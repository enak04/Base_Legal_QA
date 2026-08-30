# =============================================================================
# infer_lsim.py — LSIM Component 3: Legal In-Context Learning
# Paper : "Elevating Legal LLM Responses" (NAACL 2025)
# Section: 3.3 Legal in-context learning + Eq. 12

# Complete LSIM pipeline (all 3 components):
# Step 1 → PolicyNetwork (RL) predicts Cz_qi (fact-rule chain extension)
# Step 2 → DSSMModel retrieves top-K=3 QA pairs from database
# Step 3 → LLM generates final answer using Appendix A.3 prompt

# Evaluation metrics:
# ROUGE-1, ROUGE-2, ROUGE-L, METEOR, BERTScore, BLEU-4, Cosine Semantic Similarity
# =============================================================================

import os
import sys
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch.distributions import Categorical
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # ── model checkpoints (from train_rl.py and train_dssm.py) ──────────────
    "rl_model_path"      : "rl_policy_model.pt",
    "dssm_model_path"    : "dssm_model.pt",
    "retrieval_index"    : "dssm_retrieval_index.pt",
    "question_emb_cache" : "question_embeddings.pt",

    # ── data ──────────────────────────────────────────────────────────────────
    "test_data_path" : "ground_truth_rl_dataset.csv",
    "kg_path"        : "fact_rule_chains.py",

    # ── LLM backend ──────────────────────────────────────────────────────────
    # Options: "groq" | "ollama" | "openai" | "hf_local"
    "llm_backend"    : "openai",

    # Groq (free tier — https://console.groq.com)
    "groq_api_key"   : "Your_API_KEY",
    "groq_model"     : "llama-3.3-70b-versatile",

    # Ollama (local — run `ollama pull llama3` first)
    "ollama_model"   : "llama3",
    "ollama_url"     : "http://localhost:11434/api/generate",

    # OpenAI
    "openai_api_key" : "Your_API_KEY",
    "openai_model"   : "gpt-4o",

    # HuggingFace local (CPU-friendly small model)
    "hf_model"       : "microsoft/phi-2",

    # ── LLM generation settings (paper Section 4.3) ──────────────────────────
    "temperature"    : 0.8,
    "top_p"          : 0.9,
    "max_new_tokens" : 512,

    # ── LSIM inference settings ───────────────────────────────────────────────
    "top_k"          : 3,      # paper: K=3 retrieved cases
    "rl_test_steps"  : 3,      # paper inference steps for RL

    # ── evaluation ────────────────────────────────────────────────────────────
    "eval_samples"   : 200,      # None = evaluate all samples
    "output_path"    : "lsim_results.csv",

    # ── misc ──────────────────────────────────────────────────────────────────
    "bert_model"     : "bert-base-uncased",
    "seed"           : 42,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
}

# ── Appendix A.3 system prompt (exact from paper) ─────────────────────────────
SYSTEM_PROMPT = (
    "Your task is to provide legal advice on the user's question. "
    "I will provide you with the logical structure of the user's question, "
    "along with similar questions previously asked by other users and the "
    "responses given by real lawyers. Please use this information to "
    "generate a response to the user's question."
)

# =============================================================================
# UTILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# =============================================================================
# COMPONENT 1 — PolicyNetwork (from train_rl.py, paper Section 3.1.2)
# 768 → 516 → 256 → 128 → N | ReLU + Kaiming + Softmax
# =============================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, output_size):
        super(PolicyNetwork, self).__init__()
        self.l1  = nn.Linear(768, 516)
        self.l2  = nn.Linear(516, 256)
        self.l3  = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return F.softmax(self.out(x), dim=1)

def load_rl_model(path, device):
    ckpt       = torch.load(path, map_location=device)
    all_nodes  = ckpt["all_nodes"]
    node2idx   = ckpt["node2idx"]
    idx2node   = ckpt["idx2node"]
    policy     = PolicyNetwork(output_size=len(all_nodes)).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"RL model | Loaded best_val={ckpt['best_val_reward']:.4f}"
          f" epoch={ckpt['epoch']}")
    return policy, all_nodes, node2idx, idx2node

# =============================================================================
# COMPONENT 1 — KG graph for RL traversal (3-tier successor)
# =============================================================================
def load_kg(kg_path):
    with open(kg_path, "r", encoding="utf-8") as f:
        src = f.read()
    ns = {}
    exec(src, ns)
    G = nx.DiGraph()
    for n in ns["facts"]:         G.add_node(n, node_type="fact")
    for n in ns["rules"]:         G.add_node(n, node_type="rule")
    for s, d, r in ns["edges"]:   G.add_edge(s, d, relation=r)
    node_set = set(G.nodes())
    print(f"KG | {G.number_of_nodes()} nodes {G.number_of_edges()} edges")
    return G, node_set

def get_valid_successors(G, q_cot, visited):
    """3-tier fallback preventing empty action set (same as train_rl.py)."""
    tier1 = [n for n in G.successors(q_cot[-1]) if n not in visited]
    if tier1: return tier1
    seen, tier2 = set(), []
    for node in q_cot:
        for s in G.successors(node):
            if s not in visited and s not in seen:
                seen.add(s); tier2.append(s)
    if tier2: return tier2
    seen, tier3 = set(), []
    for node in q_cot:
        for p in G.predecessors(node):
            if p not in visited and p not in seen:
                seen.add(p); tier3.append(p)
    return tier3

def predict_chain_rl(chain_q, q_embedding, policy, G,
                     all_nodes, node2idx, idx2node, device, steps=3):
    policy.eval()
    q_cot   = copy.deepcopy(chain_q)
    visited = set(q_cot)
    with torch.no_grad():
        for _ in range(steps):
            successors = get_valid_successors(G, q_cot, visited)
            if not successors: break
            valid_idx = [node2idx[n] for n in successors if n in node2idx]
            if not valid_idx: break
            probs    = policy(q_embedding).squeeze()
            mask     = torch.zeros(len(all_nodes), device=device)
            mask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = 1.0
            filtered = probs * mask
            total    = filtered.sum()
            if total.item() == 0: break
            filtered /= total
            action    = Categorical(filtered).sample()
            next_node = idx2node[action.item()]
            q_cot.append(next_node)
            visited.add(next_node)
    return q_cot

# =============================================================================
# COMPONENT 2 — DSSMModel (from train_dssm.py, paper Section 3.2)
# 1536 → 600 → 300 → 128 → 1 | Xavier Uniform
# =============================================================================
class DSSMModel(nn.Module):
    def __init__(self):
        super(DSSMModel, self).__init__()
        self.l1  = nn.Linear(768 * 2, 600)
        self.l2  = nn.Linear(600, 300)
        self.l3  = nn.Linear(300, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x1 = F.tanh(self.l1(x))
        x2 = F.tanh(self.l2(x1))
        x3 = F.tanh(self.l3(x2))
        return self.out(x3).squeeze(-1)

def load_dssm_model(path, device):
    ckpt = torch.load(path, map_location=device)
    dssm = DSSMModel().to(device)
    dssm.load_state_dict(ckpt["model_state"])
    dssm.eval()
    print(f"DSSM model | Loaded best_val_acc={ckpt['best_val_acc']*100:.1f}%"
          f" epoch={ckpt['epoch']}")
    return dssm

def retrieve_top_k(q_emb, dssm, db_h_q, db_data, device, k=3):
    M, scores = len(db_data), []
    dssm.eval()
    with torch.no_grad():
        BATCH = 64
        for start in range(0, M, BATCH):
            end     = min(start + BATCH, M)
            db_embs = db_h_q[start:end].to(device)
            q_rep   = q_emb.expand(end - start, -1).to(device)
            state   = torch.cat([q_rep, db_embs], dim=1)
            scores.append(dssm(state).cpu())
    scores = torch.cat(scores)
    topk_vals, topk_idxs = scores.topk(k)
    return [dict(**db_data[idx], score=round(v.item(), 4))
            for v, idx in zip(topk_vals, topk_idxs.tolist())]

# =============================================================================
# COMPONENT 3 — Prompt builder (Appendix A.3 exact format, Eq. 11-12)
# =============================================================================
def build_prompt(question, chain_q_extended, top_k_results):
    chain_str = " → ".join(chain_q_extended) if chain_q_extended else "N/A"
    context_lines = []
    for i, r in enumerate(top_k_results, 1):
        context_lines.append(f"Similar Question {i}: {r['question']}")
        context_lines.append(f"Lawyer's Response {i}: {r['answer']}")
        context_lines.append("")
    context_str  = "\n".join(context_lines).strip()
    user_message = f"""Logical structure of the user's question:
{chain_str}

Similar questions and lawyer responses:
{context_str}

User's question:
{question}"""
    return user_message

# =============================================================================
# LLM BACKENDS
# =============================================================================
def call_groq(prompt, config):
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")
    client   = Groq(api_key=config["groq_api_key"])
    response = client.chat.completions.create(
        model       = config["groq_model"],
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = config["temperature"],
        top_p       = config["top_p"],
        max_tokens  = config["max_new_tokens"],
    )
    return response.choices[0].message.content.strip()

def call_ollama(prompt, config):
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    response = requests.post(
        config["ollama_url"],
        json={
            "model" : config["ollama_model"],
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config["temperature"],
                "top_p"      : config["top_p"],
                "num_predict": config["max_new_tokens"],
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def call_openai(prompt, config):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    client   = OpenAI(api_key=config["openai_api_key"])
    response = client.chat.completions.create(
        model       = config["openai_model"],
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = config["temperature"],
        top_p       = config["top_p"],
        max_tokens  = config["max_new_tokens"],
    )
    return response.choices[0].message.content.strip()

def call_hf_local(prompt, config):
    from transformers import pipeline
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nAnswer:"
    gen = pipeline(
        "text-generation",
        model       = config["hf_model"],
        device      = -1,
        torch_dtype = torch.float32,
    )
    out = gen(
        full_prompt,
        max_new_tokens = config["max_new_tokens"],
        temperature    = config["temperature"],
        do_sample      = True,
        top_p          = config["top_p"],
        pad_token_id   = 50256,
    )
    generated = out[0]["generated_text"]
    if full_prompt in generated:
        generated = generated[len(full_prompt):].strip()
    return generated

def generate_answer(prompt, config):
    backend = config["llm_backend"]
    if   backend == "groq"    : return call_groq(prompt, config)
    elif backend == "ollama"  : return call_ollama(prompt, config)
    elif backend == "openai"  : return call_openai(prompt, config)
    elif backend == "hf_local": return call_hf_local(prompt, config)
    else:
        raise ValueError(f"Unknown llm_backend: {backend}. "
                         f"Choose: groq | ollama | openai | hf_local")

# =============================================================================
# BERT ENCODER (shared by RL and DSSM)
# =============================================================================
def load_bert(model_name, device):
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()
    return tok, bert

def encode_text(text, tok, bert, device, max_len=128):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True).to(device)
        cls = bert(**enc).last_hidden_state[:, 0, :]  # (1, 768)
    return cls

# =============================================================================
# EVALUATION
# Metrics: ROUGE-1/2/L, METEOR, BERTScore, BLEU-4, Cosine Semantic Similarity
#
# BLEU-4 : corpus-level 4-gram BLEU via sacrebleu class-based API
#          (sacrebleu.metrics.BLEU) — works across all sacrebleu versions.
#          max_ngram_order=4, smooth_method="exp" (exponential smoothing).
# Cosine : mean cosine similarity between BERT [CLS] or SBERT embeddings
#          → measures semantic similarity beyond surface n-gram overlap
# =============================================================================
def load_evaluators():
    """Load all evaluation metric libraries."""
    evals = {}

    # ROUGE
    try:
        from rouge import Rouge
        evals["rouge"] = Rouge()
        print("Metrics | ROUGE loaded ✓")
    except ImportError:
        print("Metrics | ROUGE not available — pip install rouge")

    # METEOR
    try:
        import nltk
        nltk.download("wordnet",                    quiet=True)
        nltk.download("punkt",                      quiet=True)
        nltk.download("punkt_tab",                  quiet=True)
        nltk.download("omw-1.4",                    quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        evals["meteor"] = nltk_meteor
        print("Metrics | METEOR loaded ✓")
    except ImportError:
        print("Metrics | METEOR not available — pip install nltk")

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        evals["bert_score"] = bert_score_fn
        print("Metrics | BERTScore loaded ✓")
    except ImportError:
        print("Metrics | BERTScore not available — pip install bert-score")

    # BLEU-4 — use class-based API (works on all sacrebleu versions)
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        evals["bleu"] = SacreBLEU
        print("Metrics | BLEU-4 (sacrebleu) loaded ✓")
    except ImportError:
        print("Metrics | BLEU-4 not available — pip install sacrebleu")

    # Cosine Semantic Similarity
    try:
        from sentence_transformers import SentenceTransformer
        evals["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
        print("Metrics | Cosine Sim (SBERT all-MiniLM-L6-v2) loaded ✓")
    except ImportError:
        evals["cosine_fallback"] = True
        print("Metrics | Cosine Sim will use BERT CLS embeddings "
              "(pip install sentence-transformers for faster/better SBERT)")

    return evals


def compute_cosine_similarity(predictions, references, evals,
                               bert_embeddings=None):
    """
    Compute mean cosine similarity between prediction and reference embeddings.
    Priority:
      1. sentence-transformers SBERT (best quality, fast)
      2. BERT CLS embeddings pre-computed during inference (fallback)
    Returns float score in [0, 100].
    """
    if "sbert" in evals:
        model  = evals["sbert"]
        p_embs = model.encode(predictions, convert_to_tensor=True,
                               show_progress_bar=False, normalize_embeddings=True)
        r_embs = model.encode(references,  convert_to_tensor=True,
                               show_progress_bar=False, normalize_embeddings=True)
        sims   = (p_embs * r_embs).sum(dim=1)
        return float(sims.mean().item()) * 100

    elif bert_embeddings is not None:
        p_embs = bert_embeddings["pred"]
        r_embs = bert_embeddings["ref"]
        p_norm = F.normalize(p_embs, dim=1)
        r_norm = F.normalize(r_embs, dim=1)
        sims   = (p_norm * r_norm).sum(dim=1)
        return float(sims.mean().item()) * 100

    return None


def compute_metrics(predictions, references, evals, bert_embeddings=None):
    """
    Compute all evaluation metrics.
    Returns dict: metric_name → mean_value (all in 0-100 range).

    Metrics:
      rouge_1, rouge_2, rouge_l  — n-gram overlap (ROUGE F1 %)
      meteor                     — recall-focused + synonym matching (%)
      bert_score                 — BERT contextual similarity F1 (%)
      bleu_4                     — corpus-level BLEU-4 via sacrebleu class API,
                                   max_ngram_order=4, smooth_method="exp" (0-100)
      cosine_sim                 — mean BERT/SBERT cosine similarity (%)
    """
    results = {}

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    if "rouge" in evals:
        try:
            rouge_scores       = evals["rouge"].get_scores(
                predictions, references, avg=True)
            results["rouge_1"] = rouge_scores["rouge-1"]["f"] * 100
            results["rouge_2"] = rouge_scores["rouge-2"]["f"] * 100
            results["rouge_l"] = rouge_scores["rouge-l"]["f"] * 100
        except Exception as e:
            print(f"  ROUGE error: {e}")

    # ── METEOR ────────────────────────────────────────────────────────────────
    if "meteor" in evals:
        try:
            import nltk
            scores = []
            for pred, ref in zip(predictions, references):
                ref_tok  = nltk.word_tokenize(ref.lower())
                pred_tok = nltk.word_tokenize(pred.lower())
                scores.append(evals["meteor"]([ref_tok], pred_tok))
            results["meteor"] = float(np.mean(scores)) * 100
        except Exception as e:
            print(f"  METEOR error: {e}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    if "bert_score" in evals:
        try:
            P, R, F1 = evals["bert_score"](
                predictions, references, lang="en", verbose=False)
            results["bert_score"] = F1.mean().item() * 100
        except Exception as e:
            print(f"  BERTScore error: {e}")

    # ── BLEU-4 (sacrebleu class-based API — version-independent) ─────────────
    # Uses sacrebleu.metrics.BLEU class instead of corpus_bleu() function.
    # max_ngram_order=4  → cumulative BLEU-4 (1+2+3+4-gram, equal weights)
    # smooth_method="exp" → exponential smoothing prevents zero scores when
    #                        4-gram matches are sparse (standard for long text)
    if "bleu" in evals:
        try:
            bleu_scorer = evals["bleu"](
                max_ngram_order = 4,
                smooth_method   = "exp",
            )
            bleu_result      = bleu_scorer.corpus_score(predictions, [references])
            results["bleu_4"] = bleu_result.score   # already in 0-100
        except Exception as e:
            print(f"  BLEU-4 error: {e}")

    # ── Cosine Semantic Similarity ────────────────────────────────────────────
    try:
        cos_val = compute_cosine_similarity(
            predictions, references, evals, bert_embeddings)
        if cos_val is not None:
            results["cosine_sim"] = cos_val
    except Exception as e:
        print(f"  Cosine Sim error: {e}")

    return results

# =============================================================================
# FULL LSIM PIPELINE (Eq. 12: a'i = LLM(qi, Cqi, context_i))
# =============================================================================
def run_lsim(config):
    set_seed(config["seed"])
    device = torch.device(config["device"])
    print(f"\nDevice     : {device}")
    print(f"LLM Backend: {config['llm_backend']}\n{'='*66}")

    # ── Load all models ────────────────────────────────────────────────────────
    G, node_set = load_kg(config["kg_path"])

    policy, all_nodes, node2idx, idx2node = load_rl_model(
        config["rl_model_path"], device)

    dssm = load_dssm_model(config["dssm_model_path"], device)

    index   = torch.load(config["retrieval_index"], map_location=device)
    db_h_q  = index["h_q"].to(device)
    db_data = index["db_data"]
    print(f"Retrieval  | {len(db_data)} database entries loaded")

    tok, bert = load_bert(config["bert_model"], device)

    # ── Load test data ─────────────────────────────────────────────────────────
    df = pd.read_csv(config["test_data_path"])
    test_samples = []
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
        if not cq: continue
        test_samples.append({
            "question"  : str(row["question"]),
            "answer_ref": str(row["answer"]),
            "chain_q"   : cq,
        })

    n = config["eval_samples"]
    test_samples = test_samples if n is None else test_samples[:n]
    print(f"Test set   | {len(test_samples)} samples")
    print(f"{'='*66}\n")

    # ── Load evaluators ────────────────────────────────────────────────────────
    evals = load_evaluators()

    # ── Run inference loop ─────────────────────────────────────────────────────
    records        = []
    predictions    = []
    references     = []
    pred_bert_embs = []
    ref_bert_embs  = []

    for i, sample in enumerate(tqdm(test_samples,
                                    desc="LSIM Inference", unit="sample")):
        question   = sample["question"]
        chain_q    = sample["chain_q"]
        answer_ref = sample["answer_ref"]

        # Step 1: RL → extend chain_q to Cz_qi
        q_emb          = encode_text(question, tok, bert, device)
        chain_extended = predict_chain_rl(
            chain_q, q_emb, policy, G,
            all_nodes, node2idx, idx2node,
            device, steps=config["rl_test_steps"])

        # Step 2: DSSM → retrieve top-K=3 QA pairs
        top_k = retrieve_top_k(q_emb, dssm, db_h_q, db_data, device,
                                k=config["top_k"])

        # Step 3: Build prompt (Appendix A.3)
        prompt = build_prompt(question, chain_extended, top_k)

        # Step 4: LLM generates answer
        try:
            answer_pred = generate_answer(prompt, config)
        except Exception as e:
            print(f"\n  Sample {i+1} LLM error: {e}")
            answer_pred = "[LLM_ERROR]"

        predictions.append(answer_pred)
        references.append(answer_ref)

        # Collect BERT CLS embeddings for cosine fallback
        if "cosine_fallback" in evals and answer_pred != "[LLM_ERROR]":
            with torch.no_grad():
                pred_bert_embs.append(
                    encode_text(answer_pred, tok, bert, device).squeeze(0).cpu())
                ref_bert_embs.append(
                    encode_text(answer_ref,  tok, bert, device).squeeze(0).cpu())

        records.append({
            "sample_id"        : i + 1,
            "question"         : question,
            "chain_q_orig"     : " → ".join(chain_q),
            "chain_q_extended" : " → ".join(chain_extended),
            "retrieved_q1"     : top_k[0]["question"] if len(top_k) > 0 else "",
            "retrieved_q2"     : top_k[1]["question"] if len(top_k) > 1 else "",
            "retrieved_q3"     : top_k[2]["question"] if len(top_k) > 2 else "",
            "answer_ref"       : answer_ref,
            "answer_pred"      : answer_pred,
        })

        if i < 3:
            print(f"\n── Sample {i+1} ──────────────────────────────────────────")
            print(f"  Question   : {question[:100]}")
            print(f"  Chain Czqi : {chain_extended}")
            print(f"  Retrieved1 : {top_k[0]['question'][:80]}")
            print(f"  Ref answer : {answer_ref[:120]}")
            print(f"  Pred answer: {answer_pred[:120]}")

    # ── Compute evaluation metrics ─────────────────────────────────────────────
    valid_preds = [p for p in predictions if p != "[LLM_ERROR]"]
    valid_refs  = [r for p, r in zip(predictions, references)
                   if p != "[LLM_ERROR]"]

    bert_embeddings = None
    if pred_bert_embs and ref_bert_embs:
        bert_embeddings = {
            "pred": torch.stack(pred_bert_embs),
            "ref" : torch.stack(ref_bert_embs),
        }

    print(f"\n{'='*66}")
    print(f"  EVALUATION RESULTS  ({len(valid_preds)}/{len(predictions)} samples)")
    print(f"{'='*66}")

    metrics = {}
    if valid_preds:
        metrics = compute_metrics(valid_preds, valid_refs, evals,
                                  bert_embeddings=bert_embeddings)
        for name, val in metrics.items():
            label = {
                "rouge_1"   : "ROUGE-1          ",
                "rouge_2"   : "ROUGE-2          ",
                "rouge_l"   : "ROUGE-L          ",
                "meteor"    : "METEOR           ",
                "bert_score": "BERTScore F1     ",
                "bleu_4"    : "BLEU-4 (corpus)  ",
                "cosine_sim": "Cosine Sim (sem) ",
            }.get(name, name.upper())
            print(f"  {label} : {val:.2f}")
        for rec in records:
            for name, val in metrics.items():
                rec[name] = round(val, 4)

    # ── Save results ───────────────────────────────────────────────────────────
    pd.DataFrame(records).to_csv(config["output_path"], index=False)
    print(f"\nResults saved → {config['output_path']}")
    return records, metrics

# =============================================================================
# SINGLE-QUESTION INTERACTIVE MODE
# =============================================================================
def answer_question(question, chain_q,
                    policy, G, all_nodes, node2idx, idx2node,
                    dssm, db_h_q, db_data,
                    tok, bert, device, config):
    q_emb          = encode_text(question, tok, bert, device)
    chain_extended = predict_chain_rl(
        chain_q, q_emb, policy, G, all_nodes, node2idx, idx2node,
        device, steps=config["rl_test_steps"])
    top_k   = retrieve_top_k(q_emb, dssm, db_h_q, db_data, device,
                              k=config["top_k"])
    prompt  = build_prompt(question, chain_extended, top_k)
    answer  = generate_answer(prompt, config)
    return {
        "question"        : question,
        "chain_q_extended": chain_extended,
        "retrieved_cases" : top_k,
        "prompt"          : prompt,
        "answer"          : answer,
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    records, metrics = run_lsim(CONFIG)
