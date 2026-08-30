#!/usr/bin/env python3
"""
build_ground_truth.py
=====================
Builds ground_truth_rl_dataset.csv from rl_training_data.csv
using the Labour Knowledge Graph and OpenAI models.
"""

import os
import sys
import json
import re
import time
import logging
import argparse
import importlib.util

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_ground_truth.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

CONFIG = {
    "input_csv": "rl_training_data.csv",
    "kg_path": "fact_rule_chains.py",
    "output_csv": "ground_truth_rl_dataset.csv",
    "checkpoint_path": "build_gt_checkpoint.csv",
    "openai_api_key": "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",
    "openai_model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 128,
    "top_k_candidates": 30,
    "max_bridge_nodes": 2,
    "max_chain_len": 4,
    "retry_attempts": 3,
    "retry_delay": 8.0,
    "call_delay": 0.2,
    "checkpoint_every": 10,
    "seed": 42,
}


def load_kg(kg_path):
    log.info("Loading KG from %s ...", kg_path)
    spec = importlib.util.spec_from_file_location("fact_rule_chains", kg_path)
    frc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frc)

    G = nx.DiGraph()
    all_node_labels = list(frc.facts) + list(frc.rules)
    for n in all_node_labels:
        G.add_node(n)
    for (src, tgt, rel) in frc.edges:
        src, tgt = str(src).strip(), str(tgt).strip()
        if src and tgt:
            G.add_edge(src, tgt, relation=str(rel).strip())

    nodes = sorted(G.nodes(), key=str)
    leaf_count = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    log.info(" Nodes : %d | Edges : %d", G.number_of_nodes(), G.number_of_edges())
    log.info(" Leaf nodes (out-degree 0): %d / %d", leaf_count, len(nodes))
    return G, nodes


def build_tfidf(nodes):
    log.info("Building TF-IDF index over %d KG nodes ...", len(nodes))
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        min_df=1,
        sublinear_tf=True,
    )
    mat = vec.fit_transform(nodes)
    log.info(" TF-IDF vocab size: %d", len(vec.vocabulary_))
    return vec, mat


def retrieve_candidates(text, nodes, vectorizer, node_matrix, top_k=30):
    query_vec = vectorizer.transform([text])
    sims = cosine_similarity(query_vec, node_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [nodes[i] for i in top_idx]


SYSTEM_PROMPT = (
    "You are a legal knowledge graph expert. "
    "Your task is to identify the most relevant nodes "
    "from a legal Knowledge Graph for a given legal text."
)


def build_prompt(legal_text, text_type, candidates):
    numbered_lines = ["{:2d}. {}".format(i + 1, node) for i, node in enumerate(candidates)]
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
    client = OpenAI(api_key=config["openai_api_key"])
    resp = client.chat.completions.create(
        model=config["openai_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return resp.choices[0].message.content.strip()


def parse_response(raw, candidates):
    cset = set(candidates)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            valid = [str(n).strip() for n in arr if str(n).strip() in cset]
            if valid:
                return valid[:4]
    except (json.JSONDecodeError, ValueError):
        pass

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

    quoted = re.findall(r'"([^"]+)"', raw)
    valid = [q.strip() for q in quoted if q.strip() in cset]
    if valid:
        return valid[:4]

    rl = raw.lower()
    fuzzy = [c for c in candidates if c.lower() in rl]
    return fuzzy[:4]


def select_nodes(legal_text, text_type, candidates, config):
    prompt = build_prompt(legal_text, text_type, candidates)
    for attempt in range(1, config["retry_attempts"] + 1):
        try:
            raw = call_llm(prompt, config)
            nodes = parse_response(raw, candidates)
            if nodes:
                return nodes
            log.warning(" Attempt %d: no valid nodes parsed. Raw: %.120s", attempt, raw)
        except Exception as exc:
            log.warning(" Attempt %d failed: %s", attempt, exc)
            if attempt < config["retry_attempts"]:
                time.sleep(config["retry_delay"])
    log.warning(" All attempts failed — using TF-IDF top-2 as fallback.")
    return candidates[:2]


def verify_and_bridge(G, selected, max_bridge=2, max_chain=4):
    if len(selected) <= 1:
        return selected
    G_ud = G.to_undirected()
    existing = [n for n in selected if G.has_node(n)]
    if not existing:
        return selected
    sub = G_ud.subgraph(existing)
    if nx.is_connected(sub):
        return existing[:max_chain]

    components = list(nx.connected_components(sub))
    bridge_nodes = []
    added = 0
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


def build_ground_truth(config):
    np.random.seed(config["seed"])
    log.info("=" * 68)
    log.info(" BUILD GROUND TRUTH DATASET — OpenAI %s", config["openai_model"])
    log.info(" OpenAI API calls enabled")
    log.info("=" * 68)

    df = pd.read_csv(config["input_csv"])
    df.columns = [c.strip().lstrip("\ufeff").lower() for c in df.columns]
    df = df[["question", "answer"]].dropna().reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), config["input_csv"])
    log.info("API calls needed : ~%d (2 per row)", len(df) * 2)

    G, nodes = load_kg(config["kg_path"])
    vec, node_mat = build_tfidf(nodes)

    results = []
    done_indices = set()
    ckpt = config["checkpoint_path"]

    if os.path.exists(ckpt):
        ck = pd.read_csv(ckpt)
        results = ck.to_dict(orient="records")
        done_indices = {r["row_idx"] for r in results}
        log.info("Resuming — %d rows already processed.", len(done_indices))
    else:
        log.info("No checkpoint found — starting fresh.")

    remaining = [i for i in range(len(df)) if i not in done_indices]
    log.info("Rows to process : %d (total: %d)", len(remaining), len(df))
    log.info("=" * 68)

    for count, idx in enumerate(tqdm(remaining, desc="Building chains", unit="row")):
        row = df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])

        cq_cands = retrieve_candidates(question, nodes, vec, node_mat, top_k=config["top_k_candidates"])
        cq_raw = select_nodes(question, "question", cq_cands, config)
        chain_q = verify_and_bridge(G, cq_raw, config["max_bridge_nodes"], config["max_chain_len"])
        time.sleep(config["call_delay"])

        ca_cands = retrieve_candidates(answer, nodes, vec, node_mat, top_k=config["top_k_candidates"])
        ca_raw = select_nodes(answer, "answer/response", ca_cands, config)
        chain_a = verify_and_bridge(G, ca_raw, config["max_bridge_nodes"], config["max_chain_len"])
        time.sleep(config["call_delay"])

        results.append({
            "row_idx": idx,
            "question": question,
            "answer": answer,
            "chain_q": json.dumps(chain_q, ensure_ascii=False),
            "chain_a": json.dumps(chain_a, ensure_ascii=False),
        })

        if (count + 1) % config["checkpoint_every"] == 0:
            pd.DataFrame(results).to_csv(ckpt, index=False, encoding="utf-8-sig")
            log.info(" Checkpoint saved — %d rows done.", len(results))

        if (count + 1) % 25 == 0:
            r = results[-1]
            log.info(" Row %4d | chain_q=%.60s | chain_a=%.60s", idx, r["chain_q"], r["chain_a"])

    out_df = pd.DataFrame(results)[["question", "answer", "chain_q", "chain_a"]]
    out_df.to_csv(config["output_csv"], index=False, encoding="utf-8-sig")

    nonempty_q = (out_df["chain_q"] != "[]").sum()
    nonempty_a = (out_df["chain_a"] != "[]").sum()
    both = ((out_df["chain_q"] != "[]") & (out_df["chain_a"] != "[]")).sum()

    log.info("=" * 68)
    log.info(" DONE -> %s", config["output_csv"])
    log.info(" Total rows : %d", len(out_df))
    log.info(" chain_q non-empty : %d / %d", nonempty_q, len(out_df))
    log.info(" chain_a non-empty : %d / %d", nonempty_a, len(out_df))
    log.info(" Both chains non-empty : %d / %d <- usable by train_rl.py", both, len(out_df))
    log.info("=" * 68)

    if os.path.exists(ckpt):
        os.remove(ckpt)
        log.info(" Checkpoint file removed.")

    return out_df


def dry_run(config, n_rows=3):
    log.info("── DRY RUN (TF-IDF only, no OpenAI calls) ─────────────")
    df = pd.read_csv(config["input_csv"])
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["question", "answer"]].dropna()

    G, nodes = load_kg(config["kg_path"])
    vec, node_mat = build_tfidf(nodes)

    for i, (_, row) in enumerate(df.head(n_rows).iterrows()):
        log.info("\nRow %d — QUESTION top-10 candidates:", i)
        cands = retrieve_candidates(row["question"], nodes, vec, node_mat, top_k=10)
        for j, n in enumerate(cands, 1):
            log.info(" %2d. %s", j, n)

        log.info("Row %d — ANSWER top-10 candidates:", i)
        cands = retrieve_candidates(row["answer"], nodes, vec, node_mat, top_k=10)
        for j, n in enumerate(cands, 1):
            log.info(" %2d. %s", j, n)

    log.info("\nDry run complete. If candidates look relevant, run the full pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build ground_truth_rl_dataset.csv from "
            "rl_training_data.csv + KG using OpenAI"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Test KG + TF-IDF pipeline without any OpenAI calls")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides CONFIG value)")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model name (default: gpt-4o-mini)")
    args = parser.parse_args()

    if args.api_key:
        CONFIG["openai_api_key"] = args.api_key
    if args.model:
        CONFIG["openai_model"] = args.model

    if args.dry_run:
        dry_run(CONFIG)
    else:
        if not CONFIG.get("openai_api_key"):
            log.error(
                "Please set your OpenAI API key:\n"
                " Option 1 — edit CONFIG[\"openai_api_key\"] in this file\n"
                " Option 2 — python build_ground_truth.py --api-key sk-xxxx\n"
            )
            sys.exit(1)
        build_ground_truth(CONFIG)
