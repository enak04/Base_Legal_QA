"""
infer_lsim_v3.py — chain-only inference for LSIM
Uses RL-generated fact-rule-chain directly with the LLM.
No retrieval is used during inference. Full evaluation metrics retained.
"""

import os
import json
import copy
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch.distributions import Categorical
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG = {
    "rl_model_path":        "rl_policy_model.pt",
    "node_emb_cache_path":  "node_embeddings.pt",
    "test_data_path":       "ground_truth_rl_dataset.csv",
    "kg_path":              "edges.csv",
    # ── RL inference ──────────────────────────────────────────────────
    "rl_test_steps":          4,
    "use_chain_query_expand": True,
    # ── LLM ───────────────────────────────────────────────────────────
    "llm_backend":   "openai",          # groq | openai | ollama | hflocal
    "groq_api_key":  os.environ.get("GROQ_API_KEY", ""),
    "groq_model":    "llama-3.3-70b-versatile",
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model":  "gpt-4o",
    "ollama_model":  "llama3",
    "ollama_url":    "http://localhost:11434/api/generate",
    "hf_model":      "microsoft/phi-2",
    "temperature":   0,
    "top_p":         0.2,
    "max_new_tokens": 512,
    # ── eval ──────────────────────────────────────────────────────────
    "eval_samples": 200,
    "output_path":  "lsim_v3_results.csv",
    # ── misc ──────────────────────────────────────────────────────────
    "bert_model": "law-ai/InLegalBERT",
    "seed":       42,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}

# SYSTEM_PROMPT = (
#     "You are an experienced criminal defense attorney. "
#     "Provide specific, accurate, actionable legal advice. "
#     "Always cite applicable law or legal doctrine. "
#     "Keep answers concise (3-5 sentences). "
#     "Do NOT add disclaimers about consulting a lawyer — the user already knows."
# )

# SYSTEM_PROMPT = (
#     "You are a restricted answer generator. "
#     "Generate the answer mainly from the provided fact-rule-chain. "
#     "Keep the answer short, plain, and close to the chain wording. "
#     "Do not add detailed legal reasoning, broad explanation, or extra knowledge unless necessary to form a sentence. "
#     "Do not refuse to answer. "
#     "Always give a short answer."
# )

SYSTEM_PROMPT = (
    "You are a restricted answer generator. "
    "Write an answer mainly from the given fact-rule-chain. "
    "Keep the answer short, plain, and conservative. "
    "Do not add detailed reasoning, broad explanation, extra legal knowledge, or polished advice. "
    "Use simple wording and stay close to the chain content. "
    "Always answer in 2 or 3 short sentences."
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_kg(kg_path):
    G = nx.DiGraph()
    ext = os.path.splitext(kg_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(kg_path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            s = str(row["source"]).strip()
            t = str(row["target"]).strip()
            if s in ("source", "nan", "") or t in ("target", "nan", ""):
                continue
            G.add_edge(s, t, relation=str(row.get("relation", "RELATED")).strip())
    elif ext == ".pkl":
        import pickle
        with open(kg_path, "rb") as f:
            raw = pickle.load(f)
        G = nx.DiGraph(raw) if not isinstance(raw, nx.DiGraph) else raw
    node_set = set(G.nodes)
    print(f"KG | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G, node_set


def load_bert(model_name, device):
    tok = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()
    return tok, bert


def encode_text(text, tok, bert, device, max_len=128):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True).to(device)
        return bert(**enc).last_hidden_state[:, 0, :]


def encode_text_expanded(question, chain_nodes, tok, bert, device,
                          max_len=128, use_expand=True):
    if use_expand and chain_nodes:
        q_lower = question.lower()
        extras = [n for n in chain_nodes if n.lower() not in q_lower][:4]
        text = question + (" [" + " | ".join(extras) + "]" if extras else "")
    else:
        text = question
    return encode_text(text, tok, bert, device, max_len)


# ─────────────────────────────────────────────────────────────────────────────
# Model classes
# ─────────────────────────────────────────────────────────────────────────────
class ChainAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn_proj = nn.Linear(dim, 1, bias=False)

    def forward(self, node_embs_list, fallback_emb):
        if not node_embs_list:
            return fallback_emb
        stacked = torch.stack(node_embs_list, dim=0)
        weights = torch.softmax(self.attn_proj(stacked).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(0)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        r = x
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return F.relu(self.norm(x + r))


class PolicyNetwork(nn.Module):
    def __init__(self, output_size, input_dim=768 * 2, hidden=768, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.res1 = ResidualBlock(hidden, dropout)
        self.res2 = ResidualBlock(hidden, dropout)
        self.neck = nn.Linear(hidden, 256)
        self.out  = nn.Linear(256, output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.proj(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.drop(F.relu(self.neck(x)))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (RL only — DSSM not needed)
# ─────────────────────────────────────────────────────────────────────────────
def load_rl_model(path, device):
    ckpt = torch.load(path, map_location=device)
    all_nodes = ckpt["all_nodes"]
    node2idx  = ckpt["node2idx"]
    idx2node  = ckpt["idx2node"]
    policy = PolicyNetwork(output_size=len(all_nodes), input_dim=768 * 2).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    chain_attn = ChainAttention(dim=768).to(device)
    if "chain_attn_state_dict" in ckpt:
        chain_attn.load_state_dict(ckpt["chain_attn_state_dict"])
    policy.eval()
    chain_attn.eval()
    print(f"RL model loaded | epoch={ckpt.get('epoch','NA')} | best_val={ckpt.get('best_val_reward',0):.4f}")
    return policy, chain_attn, all_nodes, node2idx, idx2node


# ─────────────────────────────────────────────────────────────────────────────
# KG traversal
# ─────────────────────────────────────────────────────────────────────────────
def get_valid_successors(G, q_cot, visited):
    tier1 = [n for n in G.successors(q_cot[-1]) if n not in visited]
    if tier1:
        return tier1
    seen, tier2 = set(), []
    for node in q_cot:
        for s in G.successors(node):
            if s not in visited and s not in seen:
                seen.add(s); tier2.append(s)
    if tier2:
        return tier2
    seen, tier3 = set(), []
    for node in q_cot:
        for p in G.predecessors(node):
            if p not in visited and p not in seen:
                seen.add(p); tier3.append(p)
    return tier3


def get_chain_state(chain_nodes, node_emb_dict, fallback_emb, chain_attn, device):
    valid = [node_emb_dict[n].to(device) for n in chain_nodes if n in node_emb_dict]
    return chain_attn(valid, fallback_emb)


# ─────────────────────────────────────────────────────────────────────────────
# RL chain prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_chain_rl(chain_q, q_emb, policy, chain_attn, G, all_nodes,
                     node2idx, idx2node, node_emb_dict, device, steps=4):
    policy.eval(); chain_attn.eval()
    q_cot   = copy.deepcopy(chain_q)
    visited = set(q_cot)
    q_emb_d = q_emb.squeeze(0)
    with torch.no_grad():
        for _ in range(steps):
            succs = get_valid_successors(G, q_cot, visited)
            if not succs:
                break
            valid_idx = [node2idx[n] for n in succs if n in node2idx]
            if not valid_idx:
                break
            chain_state = get_chain_state(q_cot, node_emb_dict, q_emb_d, chain_attn, device)
            state_emb   = torch.cat([q_emb_d, chain_state], dim=0).unsqueeze(0)
            logits      = policy(state_emb).squeeze(0)
            mask        = torch.full((len(all_nodes),), float("-inf"), device=device)
            mask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = 0.0
            probs  = F.softmax(logits + mask, dim=0)
            action = Categorical(probs).sample()
            nxt    = idx2node[action.item()]
            q_cot.append(nxt); visited.add(nxt)
    return q_cot


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — chain only (no retrieval)
# ─────────────────────────────────────────────────────────────────────────────
# def build_prompt(question, chain_q_extended):
#     chain_str = " -> ".join(chain_q_extended) if chain_q_extended else "N/A"
#     return f"""You are advising on a criminal law question. Use only the legal reasoning chain below.

# LEGAL REASONING CHAIN (key facts and applicable rules):
# {chain_str}

# USER'S QUESTION:
# {question}

# Instructions:
# - Identify the specific legal issue and applicable statute or doctrine from the chain.
# - Directly address the user's exact situation (not a generic answer).
# - State what the user CAN and CANNOT do legally.
# - Give 1-2 concrete next steps.
# - Be concise (3-5 sentences). Do not begin with "I" or "As an AI".
# - Start your answer with the most important legal point.

# ANSWER:"""

# def build_prompt(question, chain_q_extended):
#     chain_str = " -> ".join(chain_q_extended) if chain_q_extended else "N/A"
#     return f"""Answer the question using mainly the fact-rule-chain below.

# FACT-RULE-CHAIN:
# {chain_str}

# QUESTION:
# {question}

# Rules:
# - Base the answer mainly on the fact-rule-chain.
# - Keep the answer short and simple.
# - Do not add much extra reasoning.
# - Do not give a detailed legal explanation.
# - Do not refuse.
# - Write 2 to 4 short sentences.
# - Use plain wording close to the chain content.

# ANSWER:"""

def build_prompt(question, chain_q_extended):
    chain_str = " -> ".join(chain_q_extended) if chain_q_extended else "N/A"
    return f"""Answer the question using mainly the fact-rule-chain.

FACT-RULE-CHAIN:
{chain_str}

QUESTION:
{question}

Rules:
- Use mainly the fact-rule-chain.
- Keep the answer short and plain.
- Do not give detailed legal explanation.
- Do not expand with outside knowledge unless needed to connect the sentence.
- Prefer generic wording over specific legal advice.
- Keep some ambiguity if the chain is unclear.
- Write only 2 or 3 short sentences.

ANSWER:"""


# ─────────────────────────────────────────────────────────────────────────────
# Answer post-processing
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_answer(text):
    text = re.sub(r"^ANSWER:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^Dear (Client|User|Sir|Ma'am)[,.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^As (a |an )?(legal |licensed |experienced )?.*?,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM backends
# ─────────────────────────────────────────────────────────────────────────────
def call_groq(prompt, config):
    from groq import Groq
    client = Groq(api_key=config["groq_api_key"])
    resp = client.chat.completions.create(
        model=config["groq_model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
    )
    return resp.choices[0].message.content.strip()


def call_openai(prompt, config):
    from openai import OpenAI
    client = OpenAI(api_key=config["openai_api_key"])
    resp = client.chat.completions.create(
        model=config["openai_model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_new_tokens"],
    )
    return resp.choices[0].message.content.strip()


def call_ollama(prompt, config):
    import requests
    full = f"{SYSTEM_PROMPT}\n\n{prompt}"
    resp = requests.post(
        config["ollama_url"],
        json={
            "model":  config["ollama_model"],
            "prompt": full,
            "stream": False,
            "options": {
                "temperature": config["temperature"],
                "top_p":       config["top_p"],
                "num_predict": config["max_new_tokens"],
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def call_hf_local(prompt, config):
    from transformers import pipeline
    full = f"{SYSTEM_PROMPT}\n\n{prompt}"
    gen = pipeline("text-generation", model=config["hf_model"],
                   device=-1, torch_dtype=torch.float32)
    out = gen(full, max_new_tokens=config["max_new_tokens"],
              temperature=config["temperature"], do_sample=True,
              top_p=config["top_p"], pad_token_id=50256)
    generated = out[0]["generated_text"]
    if full in generated:
        generated = generated[len(full):].strip()
    return generated


def generate_answer(prompt, config):
    backend = config["llm_backend"]
    if backend == "groq":    return call_groq(prompt, config)
    if backend == "openai":  return call_openai(prompt, config)
    if backend == "ollama":  return call_ollama(prompt, config)
    if backend == "hflocal": return call_hf_local(prompt, config)
    raise ValueError(f"Unknown llm_backend: {backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────
def load_evaluators():
    evals = {}

    # ROUGE
    try:
        from rouge import Rouge
        evals["rouge"] = Rouge()
        print("Evaluator loaded: ROUGE")
    except Exception as e:
        print(f"WARNING: ROUGE not available: {e}")

    # METEOR
    try:
        import nltk
        for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
            nltk.download(pkg, quiet=True)
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        _t = meteor_score([word_tokenize("test sentence")], word_tokenize("test sentence"))
        assert _t > 0.0
        evals["meteor"]        = meteor_score
        evals["word_tokenize"] = word_tokenize
        print("Evaluator loaded: METEOR (NLTK)")
    except Exception as e:
        print(f"WARNING: NLTK METEOR not available ({e}). Trying sacrebleu fallback...")
        try:
            from sacrebleu.metrics import METEOR as SacreMETEOR
            evals["sacrebleu_meteor"] = SacreMETEOR()
            print("Evaluator loaded: METEOR (sacrebleu fallback)")
        except Exception as e2:
            print(f"WARNING: sacrebleu METEOR also unavailable: {e2}. METEOR will NOT be computed.")

    # BLEU + chrF
    try:
        from sacrebleu.metrics import BLEU, CHRF
        evals["sacrebleu_bleu"] = BLEU(effective_order=True)
        evals["sacrebleu_chrf"] = CHRF()
        print("Evaluator loaded: BLEU + chrF (sacrebleu)")
    except Exception as e:
        print(f"WARNING: sacrebleu BLEU not available ({e}). Trying nltk fallback...")
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            evals["nltk_bleu"]    = sentence_bleu
            evals["nltk_bleu_sf"] = SmoothingFunction().method1
            print("Evaluator loaded: BLEU (NLTK sentence_bleu fallback)")
        except Exception as e2:
            print(f"WARNING: NLTK BLEU also unavailable: {e2}. BLEU will NOT be computed.")

    # Cosine Semantic Similarity
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
        assert _sbert.encode(["test"], convert_to_tensor=True).shape[-1] > 0
        evals["sbert_model"] = _sbert
        evals["sbert_util"]  = st_util
        print("Evaluator loaded: Cosine Semantic Similarity (sentence-transformers)")
    except Exception as e:
        print(f"WARNING: sentence-transformers not available ({e}). Trying TF-IDF cosine fallback...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
            evals["tfidf_cosine"] = (TfidfVectorizer, sk_cosine)
            print("Evaluator loaded: Cosine Similarity (TF-IDF sklearn fallback)")
        except Exception as e2:
            print(f"WARNING: sklearn also unavailable: {e2}. Cosine Similarity will NOT be computed.")

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        evals["bertscore"] = bert_score_fn
        print("Evaluator loaded: BERTScore")
    except Exception as e:
        print(f"WARNING: bert_score not available: {e}")

    return evals


def compute_metrics(predictions, references, evals):
    results = {}
    if not predictions:
        print("WARNING: compute_metrics called with empty predictions list.")
        return results

    # ROUGE
    if "rouge" in evals:
        try:
            preds_safe = [p if p.strip() else "empty" for p in predictions]
            refs_safe  = [r if r.strip() else "empty" for r in references]
            rs = evals["rouge"].get_scores(preds_safe, refs_safe, avg=True)
            results["rouge1"] = rs["rouge-1"]["f"] * 100
            results["rouge2"] = rs["rouge-2"]["f"] * 100
            results["rougel"] = rs["rouge-l"]["f"] * 100
        except Exception as e:
            print(f"ERROR computing ROUGE: {e}")

    # METEOR
    if "meteor" in evals and "word_tokenize" in evals:
        try:
            meteor_fn = evals["meteor"]
            tok_fn    = evals["word_tokenize"]
            vals = []
            for pred, ref in zip(predictions, references):
                ref_tok  = tok_fn(ref.lower())  if ref.strip()  else ["empty"]
                pred_tok = tok_fn(pred.lower()) if pred.strip() else ["empty"]
                vals.append(float(meteor_fn([ref_tok], pred_tok)))
            results["meteor"] = float(np.mean(vals)) * 100
            print(f"METEOR computed over {len(vals)} samples: {results['meteor']:.2f}%")
        except Exception as e:
            print(f"ERROR computing METEOR (NLTK): {e}")
    elif "sacrebleu_meteor" in evals:
        try:
            res = evals["sacrebleu_meteor"].corpus_score(predictions, [references])
            results["meteor"] = float(res.score)
            print(f"METEOR (sacrebleu) computed: {results['meteor']:.2f}%")
        except Exception as e:
            print(f"ERROR computing METEOR (sacrebleu): {e}")
    else:
        print("WARNING: METEOR not computed — no evaluator available.")

    # BLEU + chrF
    if "sacrebleu_bleu" in evals:
        try:
            bleu_scorer  = evals["sacrebleu_bleu"]
            result_bleu  = bleu_scorer.corpus_score(predictions, [references])
            precisions   = result_bleu.precisions
            results["bleu1"]   = float(precisions[0])
            results["bleu2"]   = float(precisions[1])
            results["bleu3"]   = float(precisions[2])
            results["bleu4"]   = float(result_bleu.score)
            results["bleu_bp"] = float(result_bleu.bp)
            print(f"BLEU-4 (corpus): {results['bleu4']:.2f}  BLEU-1: {results['bleu1']:.2f}  BP: {results['bleu_bp']:.3f}")
            chrf_scorer = evals.get("sacrebleu_chrf")
            if chrf_scorer:
                result_chrf    = chrf_scorer.corpus_score(predictions, [references])
                results["chrf"] = float(result_chrf.score)
                print(f"chrF++ (corpus): {results['chrf']:.2f}")
        except Exception as e:
            print(f"ERROR computing BLEU/chrF (sacrebleu): {e}")
    elif "nltk_bleu" in evals:
        try:
            bleu_fn = evals["nltk_bleu"]
            sf      = evals["nltk_bleu_sf"]
            tok_fn  = evals.get("word_tokenize", str.split)
            vals_b1, vals_b2, vals_b4 = [], [], []
            for pred, ref in zip(predictions, references):
                ref_tok  = tok_fn(ref.lower())  if ref.strip()  else ["empty"]
                pred_tok = tok_fn(pred.lower()) if pred.strip() else ["empty"]
                vals_b1.append(bleu_fn([ref_tok], pred_tok, weights=(1,0,0,0), smoothing_function=sf))
                vals_b2.append(bleu_fn([ref_tok], pred_tok, weights=(0.5,0.5,0,0), smoothing_function=sf))
                vals_b4.append(bleu_fn([ref_tok], pred_tok, weights=(0.25,)*4, smoothing_function=sf))
            results["bleu1"] = float(np.mean(vals_b1)) * 100
            results["bleu2"] = float(np.mean(vals_b2)) * 100
            results["bleu4"] = float(np.mean(vals_b4)) * 100
            print(f"BLEU-1: {results['bleu1']:.2f}%  BLEU-2: {results['bleu2']:.2f}%  BLEU-4: {results['bleu4']:.2f}% (NLTK avg)")
        except Exception as e:
            print(f"ERROR computing BLEU (NLTK): {e}")
    else:
        print("WARNING: BLEU not computed — no evaluator available.")

    # Cosine Semantic Similarity
    if "sbert_model" in evals:
        try:
            sbert   = evals["sbert_model"]
            st_util = evals["sbert_util"]
            pred_emb = sbert.encode(predictions, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
            ref_emb  = sbert.encode(references,  convert_to_tensor=True, show_progress_bar=False, batch_size=32)
            cos_scores  = st_util.cos_sim(pred_emb, ref_emb)
            pair_scores = cos_scores.diag().cpu().numpy()
            results["cosine_sim"]     = float(np.mean(pair_scores))   * 100
            results["cosine_sim_med"] = float(np.median(pair_scores)) * 100
            print(f"Cosine Sim (mean): {results['cosine_sim']:.2f}%  (median): {results['cosine_sim_med']:.2f}%")
        except Exception as e:
            print(f"ERROR computing Cosine Sim (SBERT): {e}")
    elif "tfidf_cosine" in evals:
        try:
            TfidfVec, cosine_fn = evals["tfidf_cosine"]
            tfidf    = TfidfVec().fit(predictions + references)
            pred_vec = tfidf.transform(predictions)
            ref_vec  = tfidf.transform(references)
            pair_scores = [float(cosine_fn(pred_vec[i], ref_vec[i])[0, 0]) for i in range(len(predictions))]
            results["cosine_sim"] = float(np.mean(pair_scores)) * 100
            print(f"Cosine Similarity (TF-IDF mean): {results['cosine_sim']:.2f}%")
        except Exception as e:
            print(f"ERROR computing Cosine Sim (TF-IDF): {e}")
    else:
        print("WARNING: Cosine Similarity not computed — no evaluator available.")

    # BERTScore
    if "bertscore" in evals:
        try:
            _, _, f1 = evals["bertscore"](predictions, references, lang="en", verbose=False)
            results["bertscore"] = f1.mean().item() * 100
        except Exception as e:
            print(f"ERROR computing BERTScore: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_lsim(config=CONFIG):
    set_seed(config["seed"])
    device = torch.device(config["device"])
    print(f"Device: {device}")

    G, node_set = load_kg(config["kg_path"])
    policy, chain_attn, all_nodes, node2idx, idx2node = load_rl_model(config["rl_model_path"], device)

    node_emb_dict = {}
    if os.path.exists(config["node_emb_cache_path"]):
        node_emb_dict = torch.load(config["node_emb_cache_path"], map_location="cpu")
        print(f"Node-embs | {len(node_emb_dict)} nodes cached")
    else:
        print("WARNING: node_emb_cache_path not found.")

    tok, bert = load_bert(config["bert_model"], device)

    df = pd.read_csv(config["test_data_path"])
    test_samples = []
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
        if not cq:
            continue
        test_samples.append({
            "question":   str(row["question"]),
            "answer_ref": str(row["answer"]),
            "chain_q":    cq,
        })
    n = config["eval_samples"]
    if n:
        test_samples = test_samples[:n]
    print(f"Test samples: {len(test_samples)}")

    evals = load_evaluators()
    records = []
    predictions, references = [], []

    for i, sample in enumerate(tqdm(test_samples, desc="LSIM chain-only Inference")):
        question   = sample["question"]
        chain_q    = sample["chain_q"]
        answer_ref = sample["answer_ref"]

        # Step 1: encode question
        q_emb = encode_text_expanded(
            question, chain_q, tok, bert, device,
            use_expand=config.get("use_chain_query_expand", True)
        )

        # Step 2: RL chain extension
        chain_extended = predict_chain_rl(
            chain_q, q_emb, policy, chain_attn, G, all_nodes,
            node2idx, idx2node, node_emb_dict, device,
            steps=config["rl_test_steps"]
        )

        # Step 3: build prompt (chain only) and call LLM
        prompt = build_prompt(question, chain_extended)
        try:
            answer_pred = generate_answer(prompt, config)
            answer_pred = postprocess_answer(answer_pred)
        except Exception as e:
            print(f"[{i + 1}] LLM error: {e}")
            answer_pred = "LLM_ERROR"

        predictions.append(answer_pred)
        references.append(answer_ref)
        records.append({
            "sample_id":        i + 1,
            "question":         question,
            "chain_q_orig":     " -> ".join(chain_q),
            "chain_q_extended": " -> ".join(chain_extended),
            "answer_ref":       answer_ref,
            "answer_pred":      answer_pred,
        })

    # Compute all metrics on valid predictions
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p != "LLM_ERROR"]
    vp = [p for p, _ in valid_pairs]
    vr = [r for _, r in valid_pairs]
    metrics = compute_metrics(vp, vr, evals) if vp else {}

    # Per-sample BLEU-4
    if "nltk_bleu" in evals:
        try:
            _tok = evals.get("word_tokenize", str.split)
            _sf  = evals["nltk_bleu_sf"]
            _fn  = evals["nltk_bleu"]
            for rec in records:
                pp = rec.get("answer_pred", "")
                rr = rec.get("answer_ref",  "")
                rt = _tok(rr.lower()) if rr.strip() else ["empty"]
                pt = _tok(pp.lower()) if pp.strip() else ["empty"]
                rec["bleu4_sample"] = round(float(_fn([rt], pt, weights=(0.25,)*4, smoothing_function=_sf)) * 100, 4)
        except Exception as _e:
            print(f"WARNING: per-sample BLEU-4 failed: {_e}")
    elif "sacrebleu_bleu" in evals:
        try:
            from sacrebleu.metrics import BLEU as _SB
            _sb = _SB(effective_order=True)
            for rec in records:
                pp = rec.get("answer_pred", "")
                rr = rec.get("answer_ref",  "")
                rec["bleu4_sample"] = round(float(_sb.sentence_score(pp, [rr]).score), 4)
        except Exception as _e:
            print(f"WARNING: per-sample BLEU-4 (sacrebleu) failed: {_e}")

    # Per-sample Cosine Similarity
    if "sbert_model" in evals:
        try:
            _sbert   = evals["sbert_model"]
            _st_util = evals["sbert_util"]
            _ps = [rec.get("answer_pred", "") for rec in records]
            _rs = [rec.get("answer_ref",  "") for rec in records]
            _pe = _sbert.encode(_ps, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
            _re = _sbert.encode(_rs, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
            _cs = _st_util.cos_sim(_pe, _re).diag().cpu().numpy()
            for rec, score in zip(records, _cs):
                rec["cosine_sim_sample"] = round(float(score) * 100, 4)
        except Exception as _e:
            print(f"WARNING: per-sample cosine sim (SBERT) failed: {_e}")
    elif "tfidf_cosine" in evals:
        try:
            TfidfVec, cosine_fn = evals["tfidf_cosine"]
            _ps = [rec.get("answer_pred", "") for rec in records]
            _rs = [rec.get("answer_ref",  "") for rec in records]
            _tfidf = TfidfVec().fit(_ps + _rs)
            _pv, _rv = _tfidf.transform(_ps), _tfidf.transform(_rs)
            for idx, rec in enumerate(records):
                rec["cosine_sim_sample"] = round(float(cosine_fn(_pv[idx], _rv[idx])[0, 0]) * 100, 4)
        except Exception as _e:
            print(f"WARNING: per-sample TF-IDF cosine failed: {_e}")

    # Write corpus-level metrics into every row
    for rec in records:
        for name, val in metrics.items():
            rec[name] = round(val, 4)

    pd.DataFrame(records).to_csv(config["output_path"], index=False)
    print(f"\nSaved -> {config['output_path']}")

    all_metric_keys = [
        "rouge1", "rouge2", "rougel",
        "meteor",
        "bleu1", "bleu2", "bleu3", "bleu4", "bleu_bp",
        "chrf",
        "cosine_sim", "cosine_sim_med",
        "bertscore",
    ]
    section_labels = {
        "rouge1":     "--- ROUGE -----------------------",
        "meteor":     "--- METEOR ----------------------",
        "bleu1":      "--- BLEU ------------------------",
        "chrf":       "--- chrF++ ----------------------",
        "cosine_sim": "--- Cosine Semantic Similarity --",
        "bertscore":  "--- BERTScore -------------------",
    }
    print("\n" + "=" * 50)
    print(" FINAL EVALUATION METRICS")
    print("=" * 50)
    for k in all_metric_keys:
        if k in section_labels:
            print(f" {section_labels[k]}")
        if k in metrics:
            print(f" {k:>14}: {metrics[k]:.4f}%")
        else:
            print(f" {k:>14}: NOT COMPUTED")
    print("=" * 50)
    return records, metrics


if __name__ == "__main__":
    run_lsim(CONFIG)
