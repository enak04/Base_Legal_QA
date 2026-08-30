#!/usr/bin/env python3
"""
build_ground_truth.py
=====================
Builds ground_truth_rl_dataset.csv from rl_training_data.csv
using the Labour Knowledge Graph (Labour_Hkg_graph.pkl)
and Groq  Llama-3.3-70B  (FREE  —  console.groq.com/keys).

Paper prompt (Appendix A.1):
  "Please select 1 to 4 nodes from the provided graph that are most
   relevant to the legal question/answer.
   Ensure that the selected nodes are interconnected."

Pipeline per row
----------------
  1. TF-IDF pre-filter  -> top-30 candidate KG nodes
  2. Groq Llama-3.3-70B -> selects 1-4 interconnected nodes (chain_q / chain_a)
  3. Connectivity fix   -> bridges disconnected nodes via KG shortest path
  4. Checkpoint         -> resume-safe every CHECKPOINT_EVERY rows

Output columns
--------------
  question | answer | chain_q (JSON list) | chain_a (JSON list)

Usage
-----
  pip install pandas scikit-learn networkx groq tqdm
  python build_ground_truth.py --dry-run            # test KG + TF-IDF, zero API cost
  python build_ground_truth.py --api-key gsk_xxxx   # pass Groq key via CLI
  python build_ground_truth.py                      # full run (key must be in CONFIG)

"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import json
import re
import time
import pickle
import logging
import argparse

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from groq import Groq                          # pip install groq

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_ground_truth.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  — only 2 things to edit:
#    1. groq_api_key   -> paste your key from console.groq.com/keys
#    2. Everything else is already tuned for Llama-3.3-70B on Groq
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── File paths ───────────────────────────────────────────────────────
    "input_csv"        : "rl_training_data.csv",
    "kg_path"          : "Labour_Hkg_graph.pkl",
    "output_csv"       : "ground_truth_rl_dataset.csv",
    "checkpoint_path"  : "build_gt_checkpoint.csv",

    # ── Groq / Llama-3.3-70B  (FREE) ─────────────────────────────────────
    # Get your free key at: https://console.groq.com/keys
    "groq_api_key"     : "Your_API_KEY",
    "groq_model"       : "llama-3.1-8b-instant",                               #"llama-3.3-70b-versatile",
    "temperature"      : 0.1,   # low = deterministic
    "max_tokens"       : 128,   # JSON list is short   previously 256

    # ── TF-IDF pre-filtering ─────────────────────────────────────────────
    "top_k_candidates" : 30,    # nodes shown to Llama per call

    # ── Connectivity bridging ────────────────────────────────────────────
    "max_bridge_nodes" : 2,     # max intermediate bridge nodes
    "max_chain_len"    : 4,     # hard cap (paper: 1-4 nodes)

    # ── Reliability ──────────────────────────────────────────────────────
    "retry_attempts"   : 5,     # retries per LLM call on failure          3
    "retry_delay"      : 60.0,   # seconds between retries  (Groq is fast)             3.0
    "call_delay"       : 3.0,   # seconds between normal calls (Groq: 100k tok/min)       0.5
    "checkpoint_every" : 50,    # save checkpoint every N rows

    "seed"             : 42,
}


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load Knowledge Graph
# ════════════════════════════════════════════════════════════════════════════
def load_kg(kg_path):
    """
    Load Labour_Hkg_graph.pkl.
    Supports DiGraph, MultiDiGraph, Graph, MultiGraph.
    Returns (G as DiGraph, sorted list of node label strings).
    """
    log.info("Loading KG from  %s ...", kg_path)
    with open(kg_path, "rb") as fh:
        raw = pickle.load(fh)

    if isinstance(raw, (nx.MultiDiGraph, nx.MultiGraph)):
        G = nx.DiGraph(raw)
    elif isinstance(raw, nx.Graph) and not isinstance(raw, nx.DiGraph):
        G = raw.to_directed()
    else:
        G = raw

    nodes      = sorted(G.nodes(), key=str)
    leaf_count = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

    log.info(
        "  Nodes : %d  |  Edges : %d  |  Type : %s",
        G.number_of_nodes(), G.number_of_edges(), type(raw).__name__
    )
    log.info("  Leaf nodes (out-degree 0): %d / %d", leaf_count, len(nodes))
    return G, nodes


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TF-IDF index over KG node labels
# ════════════════════════════════════════════════════════════════════════════
def build_tfidf(nodes):
    """
    Fit a TF-IDF vectoriser on all KG node label strings.
    Returns (fitted vectorizer, node_matrix of shape [N, vocab]).
    """
    log.info("Building TF-IDF index over %d KG nodes ...", len(nodes))
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        min_df=1,
        sublinear_tf=True,
    )
    mat = vec.fit_transform(nodes)
    log.info("  TF-IDF vocab size: %d", len(vec.vocabulary_))
    return vec, mat


def retrieve_candidates(text, nodes, vectorizer, node_matrix, top_k=30):
    """Return top-K KG nodes most similar (TF-IDF cosine) to text."""
    query_vec = vectorizer.transform([text])
    sims      = cosine_similarity(query_vec, node_matrix).flatten()
    top_idx   = np.argsort(sims)[::-1][:top_k]
    return [nodes[i] for i in top_idx]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Llama-3.3-70B node selection  (paper prompt Appendix A.1)
# ════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a legal knowledge graph expert. "
    "Your task is to identify the most relevant nodes "
    "from a legal Knowledge Graph for a given legal text."
)


def build_prompt(legal_text, text_type, candidates):
    """
    Build the LLM prompt following paper Appendix A.1.
    text_type: 'question' or 'answer/response'.
    """
    numbered_lines = ["{:2d}. {}".format(i + 1, node)
                      for i, node in enumerate(candidates)]
    numbered_block = "\n".join(numbered_lines)

    return (
        "Here are candidate nodes from a Legal Knowledge Graph:\n\n"
        + numbered_block + "\n\n"
        + "Legal " + text_type + ":\n"
        + '"' + legal_text + '"' + "\n\n"
        + "Please select 1 to 4 nodes from the provided graph "
        + "that are most relevant to the legal " + text_type + ". "
        + "Ensure that the selected nodes are interconnected.\n\n"
        + "Rules:\n"
        + "- Choose ONLY nodes from the numbered list above.\n"
        + "- Return ONLY a valid JSON array of exact node names.\n"
        + '- Example: ["Node A", "Node B"]\n'
        + "- No explanation, no extra text."
    )


def call_llm(prompt, config):
    """
    Call Groq Llama-3.3-70B with the given prompt.
    Returns the raw string content of the model response.
    """
    client = Groq(api_key=config["groq_api_key"])
    resp   = client.chat.completions.create(
        model       = config["groq_model"],
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = config["temperature"],
        max_tokens  = config["max_tokens"],
    )
    return resp.choices[0].message.content.strip()


def parse_response(raw, candidates):
    """
    Parse LLM output into a list of valid KG node strings.
    Four strategies in order:
      1. Strict JSON parse
      2. JSON inside ```json ... ``` code block
      3. Regex: quoted strings matching a candidate exactly
      4. Fuzzy: candidate name appears as substring in output
    """
    cset = set(candidates)

    # 1. Strict JSON
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            valid = [str(n).strip() for n in arr if str(n).strip() in cset]
            if valid:
                return valid[:4]
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. JSON inside markdown code block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                valid = [str(n).strip() for n in arr if str(n).strip() in cset]
                if valid:
                    return valid[:4]
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Quoted strings that exactly match a candidate
    quoted = re.findall(r'"([^"]+)"', raw)
    valid  = [q.strip() for q in quoted if q.strip() in cset]
    if valid:
        return valid[:4]

    # 4. Fuzzy substring match
    rl    = raw.lower()
    fuzzy = [c for c in candidates if c.lower() in rl]
    return fuzzy[:4]


def select_nodes(legal_text, text_type, candidates, config):
    """
    Ask Llama-3.3-70B to select 1-4 interconnected nodes.
    Retries up to retry_attempts on failure.
    Falls back to TF-IDF top-2 if all retries fail.
    """
    prompt = build_prompt(legal_text, text_type, candidates)

    for attempt in range(1, config["retry_attempts"] + 1):
        try:
            raw   = call_llm(prompt, config)
            nodes = parse_response(raw, candidates)
            if nodes:
                return nodes
            log.warning(
                "  Attempt %d: no valid nodes parsed. Raw: %.120s",
                attempt, raw
            )
        except Exception as exc:
            log.warning("  Attempt %d failed: %s", attempt, exc)
            if attempt < config["retry_attempts"]:
                time.sleep(config["retry_delay"])

    log.warning("  All Llama attempts failed — using TF-IDF top-2 as fallback.")
    return candidates[:2]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Connectivity verification and bridging
# ════════════════════════════════════════════════════════════════════════════
def verify_and_bridge(G, selected, max_bridge=2, max_chain=4):
    """
    Check whether selected nodes form a connected subgraph (undirected).
    If disconnected, insert up to max_bridge intermediate KG nodes
    from the shortest path between disconnected components.
    Caps the final list to max_chain nodes.
    """
    if len(selected) <= 1:
        return selected

    G_ud     = G.to_undirected()
    existing = [n for n in selected if G.has_node(n)]
    if not existing:
        return selected

    sub = G_ud.subgraph(existing)
    if nx.is_connected(sub):
        return existing[:max_chain]

    components   = list(nx.connected_components(sub))
    bridge_nodes = []
    added        = 0

    for i in range(len(components) - 1):
        if added >= max_bridge:
            break
        src = next(iter(components[i]))
        dst = next(iter(components[i + 1]))
        try:
            path = nx.shortest_path(G_ud, src, dst)
            for mid in path[1:-1]:
                if mid not in existing and added < max_bridge:
                    bridge_nodes.append(mid)
                    added += 1
        except nx.NetworkXNoPath:
            pass

    return (existing + bridge_nodes)[:max_chain]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Main pipeline
# ════════════════════════════════════════════════════════════════════════════
def build_ground_truth(config):
    """
    Full pipeline: for every (question, answer) pair in input_csv,
    build chain_q and chain_a via TF-IDF + Llama-3.3-70B + connectivity check.
    Saves result to ground_truth_rl_dataset.csv.
    """
    np.random.seed(config["seed"])

    log.info("=" * 68)
    log.info("  BUILD GROUND TRUTH DATASET  —  Groq  %s", config["groq_model"])
    log.info("  FREE tier: 14,400 req/day  |  100,000 tokens/min")
    log.info("=" * 68)

    # Load input CSV
    df = pd.read_csv(config["input_csv"])
    df.columns = [c.strip().lstrip("\ufeff").lower() for c in df.columns]
    df = df[["question", "answer"]].dropna().reset_index(drop=True)
    log.info("Loaded %d rows from  %s", len(df), config["input_csv"])
    log.info("API calls needed : ~%d  (2 per row — all FREE on Groq)", len(df) * 2)

    # Load KG and TF-IDF index
    G, nodes      = load_kg(config["kg_path"])
    vec, node_mat = build_tfidf(nodes)

    # Resume from checkpoint if available
    results      = []
    done_indices = set()
    ckpt         = config["checkpoint_path"]

    if os.path.exists(ckpt):
        ck           = pd.read_csv(ckpt)
        results      = ck.to_dict(orient="records")
        done_indices = {r["row_idx"] for r in results}
        log.info("Resuming — %d rows already processed.", len(done_indices))
    else:
        log.info("No checkpoint found — starting fresh.")

    remaining = [i for i in range(len(df)) if i not in done_indices]
    log.info("Rows to process : %d  (total: %d)", len(remaining), len(df))
    log.info("=" * 68)

    # Main loop
    for count, idx in enumerate(
        tqdm(remaining, desc="Building chains", unit="row")
    ):
        row      = df.iloc[idx]
        question = str(row["question"])
        answer   = str(row["answer"])

        # chain_q  — nodes most relevant to the QUESTION
        cq_cands = retrieve_candidates(
            question, nodes, vec, node_mat,
            top_k=config["top_k_candidates"]
        )
        cq_raw  = select_nodes(question, "question", cq_cands, config)
        chain_q = verify_and_bridge(
            G, cq_raw,
            config["max_bridge_nodes"],
            config["max_chain_len"]
        )
        time.sleep(config["call_delay"])

        # chain_a  — nodes most relevant to the ANSWER
        ca_cands = retrieve_candidates(
            answer, nodes, vec, node_mat,
            top_k=config["top_k_candidates"]
        )
        ca_raw  = select_nodes(answer, "answer/response", ca_cands, config)
        chain_a = verify_and_bridge(
            G, ca_raw,
            config["max_bridge_nodes"],
            config["max_chain_len"]
        )
        time.sleep(config["call_delay"])

        results.append({
            "row_idx" : idx,
            "question": question,
            "answer"  : answer,
            "chain_q" : json.dumps(chain_q, ensure_ascii=False),
            "chain_a" : json.dumps(chain_a, ensure_ascii=False),
        })

        # Save checkpoint every N rows
        if (count + 1) % config["checkpoint_every"] == 0:
            pd.DataFrame(results).to_csv(ckpt, index=False, encoding="utf-8-sig")
            log.info("  Checkpoint saved — %d rows done.", len(results))

        # Live preview every 25 rows
        if (count + 1) % 25 == 0:
            r = results[-1]
            log.info(
                "  Row %4d | chain_q=%.60s | chain_a=%.60s",
                idx, r["chain_q"], r["chain_a"]
            )

    # Final save
    out_df = pd.DataFrame(results)[["question", "answer", "chain_q", "chain_a"]]
    out_df.to_csv(config["output_csv"], index=False, encoding="utf-8-sig")

    nonempty_q = (out_df["chain_q"] != "[]").sum()
    nonempty_a = (out_df["chain_a"] != "[]").sum()
    both       = (
        (out_df["chain_q"] != "[]") & (out_df["chain_a"] != "[]")
    ).sum()

    log.info("=" * 68)
    log.info("  DONE -> %s", config["output_csv"])
    log.info("  Total rows            : %d", len(out_df))
    log.info("  chain_q non-empty     : %d / %d", nonempty_q, len(out_df))
    log.info("  chain_a non-empty     : %d / %d", nonempty_a, len(out_df))
    log.info(
        "  Both chains non-empty : %d / %d  <- usable by train_rl.py",
        both, len(out_df)
    )
    log.info("=" * 68)

    if os.path.exists(ckpt):
        os.remove(ckpt)
        log.info("  Checkpoint file removed.")

    return out_df


# ════════════════════════════════════════════════════════════════════════════
#  DRY RUN  — test KG + TF-IDF with zero API cost
# ════════════════════════════════════════════════════════════════════════════
def dry_run(config, n_rows=3):
    """
    Validate the TF-IDF pipeline with no Llama calls and zero API cost.
    Shows the top-10 TF-IDF candidate nodes for the first n_rows samples.
    """
    log.info("── DRY RUN (TF-IDF only, no Groq/Llama calls) ─────────────")
    df = pd.read_csv(config["input_csv"])
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["question", "answer"]].dropna()

    G, nodes      = load_kg(config["kg_path"])
    vec, node_mat = build_tfidf(nodes)

    for i, (_, row) in enumerate(df.head(n_rows).iterrows()):
        log.info("\nRow %d — QUESTION top-10 candidates:", i)
        cands = retrieve_candidates(
            row["question"], nodes, vec, node_mat, top_k=10
        )
        for j, n in enumerate(cands, 1):
            log.info("  %2d. %s", j, n)

        log.info("Row %d — ANSWER top-10 candidates:", i)
        cands = retrieve_candidates(
            row["answer"], nodes, vec, node_mat, top_k=10
        )
        for j, n in enumerate(cands, 1):
            log.info("  %2d. %s", j, n)

    log.info(
        "\nDry run complete. "
        "If candidates look relevant, run the full pipeline."
    )


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build ground_truth_rl_dataset.csv from "
            "rl_training_data.csv + Labour KG  using  Groq Llama-3.3-70B  (FREE)"
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test KG + TF-IDF pipeline without any Groq/Llama calls (free)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Groq API key  (overrides CONFIG value)  —  console.groq.com/keys"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Groq model name  (default: llama-3.3-70b-versatile)"
    )
    args = parser.parse_args()

    if args.api_key:
        CONFIG["groq_api_key"] = args.api_key
    if args.model:
        CONFIG["groq_model"] = args.model

    if args.dry_run:
        dry_run(CONFIG)
    else:
        if CONFIG["groq_api_key"] == "YOUR_GROQ_API_KEY":
            log.error(
                "Please set your Groq API key:\n"
                "  Option 1 — edit CONFIG[\"groq_api_key\"] in this file\n"
                "  Option 2 — python build_ground_truth.py --api-key gsk_xxxx\n"
                "\n"
                "  Get your FREE key at:  https://console.groq.com/keys"
            )
            sys.exit(1)
        build_ground_truth(CONFIG)
