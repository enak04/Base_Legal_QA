# =============================================================================
#  train_dssm.py  —  LSIM Component 2: Supervised DSSM-powered RAG
#  Paper  : "Elevating Legal LLM Responses" (NAACL 2025)
#  GitHub : dssm.py  (reference implementation cross-checked)
#
#  Architecture EXACTLY matching dssm.py:
#    DSSM input  : cat([q_emb(768), candidate_emb(768)])  =  1536-d  ← FIX-1
#    DSSM layers : 1536 → 600 → 300 → 128 → 1             ← FIX-2
#    Weight init : Xavier Uniform (all 4 layers)           ← FIX-3
#    Activation  : F.tanh (all hidden layers)
#    Loss        : nn.MarginRankingLoss(margin=1)          ← FIX-4
#    DataLoader  : LegalDataset + DataLoader(batch=32)     ← FIX-5
#    Pos filter  : skip if max_sim <= threshold            ← FIX-6
#    Epochs      : 50  |  Adam lr=1e-4, wd=1e-3
#    Top-K RAG   : K=3
# =============================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
#  CONFIG
# =============================================================================
CONFIG = {
    # ── paths ─────────────────────────────────────────────────────────────────
    "train_data_path"  : "ground_truth_rl_dataset.csv",
    "db_data_path"     : "retrieval_database.csv",
    "kg_path"          : "fact_rule_chains.py",
    "train_emb_cache"  : "dssm_train_embeddings.pt",
    "db_emb_cache"     : "dssm_db_embeddings.pt",
    "save_path"        : "dssm_model.pt",
    "log_path"         : "dssm_training_log.csv",
    "retrieval_index"  : "dssm_retrieval_index.pt",

    # ── bert ──────────────────────────────────────────────────────────────────
    "bert_model"       : "bert-base-uncased",

    # ── paper training values  (exact match to dssm.py) ─────────────────────
    "epochs"           : 50,
    "batch_size"       : 32,          # dssm.py: batch_size=32
    "lr"               : 1e-4,        # dssm.py: lr=0.0001
    "weight_decay"     : 1e-3,        # dssm.py: weight_decay=1e-3
    "margin"           : 1.0,         # dssm.py: MarginRankingLoss(margin=1)
    "grad_clip"        : 1.0,

    # ── FIX-6: positive quality threshold ────────────────────────────────────
    # In dssm.py: sample skipped if max_LLM_score <= 2  (out of 5)
    # Proxy: BERT cosine similarity threshold (0–1 scale ≈ LLM score/5)
    # threshold=0.3 means sim >= 0.3 required (equivalent to LLM score > 1.5)
    "pos_sim_threshold": 0.3,

    # ── RAG inference ─────────────────────────────────────────────────────────
    "top_k"            : 3,           # paper: K=3

    # ── fresh start ───────────────────────────────────────────────────────────
    "force_fresh"      : True,        # set False after first successful run

    # ── misc ──────────────────────────────────────────────────────────────────
    "seed"             : 42,
    "device"           : "cuda" if torch.cuda.is_available() else "cpu",
    "val_split"        : 0.1,
}


# =============================================================================
#  UTILITIES
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
#  STEP 1  — Load KG nodes for chain extraction
# =============================================================================
def load_kg_nodes(kg_path):
    with open(kg_path, "r", encoding="utf-8") as f:
        src = f.read()
    ns = {}
    exec(src, ns)
    all_nodes = list(dict.fromkeys(ns["facts"] + ns["rules"]))
    return all_nodes, set(all_nodes)


# =============================================================================
#  STEP 2  — Load training data and retrieval database
# =============================================================================
def load_train_data(path, node_set):
    df, data = pd.read_csv(path), []
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
        data.append({
            "idx"      : len(data),
            "question" : str(row["question"]),
            "answer"   : str(row["answer"]),
            "chain_q"  : cq,
        })
    print(f"Train data  |  Loaded: {len(data)} queries")
    return data


def extract_chain_from_text(text, node_set, max_nodes=4):
    text_lower = text.lower()
    return [n for n in node_set if n.lower() in text_lower][:max_nodes]


def load_db_data(path, node_set):
    df       = pd.read_csv(path)
    has_cq   = "chain_q" in df.columns
    data     = []
    for _, row in df.iterrows():
        q, a = str(row["question"]), str(row["answer"])
        if has_cq:
            try:    cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
            except: cq = extract_chain_from_text(q, node_set)
        else:
            cq = extract_chain_from_text(q, node_set)
        data.append({"db_idx": len(data), "question": q, "answer": a, "chain_q": cq})

    cov = sum(1 for d in data if d["chain_q"])
    print(f"DB data     |  Loaded: {len(data)} candidates"
          f"  |  Chain coverage: {cov}/{len(data)} ({cov/len(data)*100:.1f}%)")
    return data


# =============================================================================
#  STEP 3  — BERT encoding
#
#  Following dssm.py exactly:
#    embeddings_by_name  stores the QUESTION BERT embedding (768-d)
#    cot_embeddings_by_name stores the CHAIN BERT embedding  (768-d)
#    DSSM input = cat(q_emb, candidate_emb) = 1536-d  ← FIX-1
#
#  Here we store BOTH q_emb and cot_emb so we can use either or combined.
# =============================================================================
def load_bert(model_name, device):
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()
    return tok, bert


def bert_encode_batch(texts, tok, bert, device, max_len=128, batch_size=32):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tok(batch, return_tensors="pt", truncation=True,
                        max_length=max_len, padding=True).to(device)
            cls   = bert(**enc).last_hidden_state[:, 0, :]
            all_embs.append(cls.cpu())
    return torch.cat(all_embs, dim=0)   # (N, 768)


def build_embeddings(data, tok, bert, device, cache_path, desc="data"):
    """
    Build and cache BERT embeddings.
    Stores BOTH:
      h_q  : BERT(question)        (N, 768)  ← matches embeddings_by_name in dssm.py
      h_c  : BERT(chain text)      (N, 768)  ← matches cot_embeddings_by_name in dssm.py
    DSSM input per sample = cat(h_q, h_q_candidate) = (1536)  ← FIX-1
    """
    if os.path.exists(cache_path):
        print(f"Embeddings  |  Loading cache ← {cache_path}")
        result = torch.load(cache_path, map_location="cpu")
        print(f"            |  {len(result['h_q'])} vectors  h_q/h_c each 768-d")
        return result

    print(f"Embeddings  |  Encoding {len(data)} {desc} entries with BERT ...")
    questions   = [d["question"] for d in data]
    chain_texts = [" ".join(d["chain_q"]) if d["chain_q"] else d["question"]
                   for d in data]

    print(f"            |  Encoding questions ...")
    h_q = bert_encode_batch(questions,   tok, bert, device)   # (N, 768)
    print(f"            |  Encoding chains ...")
    h_c = bert_encode_batch(chain_texts, tok, bert, device)   # (N, 768)

    result = {"h_q": h_q, "h_c": h_c}
    torch.save(result, cache_path)
    print(f"            |  Saved → {cache_path}")
    return result


# =============================================================================
#  STEP 4  — Build training triplets
#
#  FIX-6: Match dssm.py's quality filter:
#    dssm.py skips samples where max_LLM_score <= 2  (out of 5)
#    We skip samples where max BERT cos_sim <= pos_sim_threshold (default 0.3)
#    This ensures only HIGH-CONFIDENCE positive candidates are trained on.
#
#  Positive (c+) = DB candidate with HIGHEST cosine similarity to query
#  Negative (c-) = DB candidate with LOWEST  cosine similarity to query
# =============================================================================
def build_triplets(train_embs, db_embs, config):
    threshold = config["pos_sim_threshold"]
    print(f"Triplets    |  Building triplets  (threshold={threshold}) ...")

    q_norm  = F.normalize(train_embs["h_q"], dim=1)   # (N_train, 768)
    db_norm = F.normalize(db_embs["h_q"],    dim=1)   # (N_db, 768)
    sim_mat = torch.mm(q_norm, db_norm.T)              # (N_train, N_db)

    triplets = []
    skipped  = 0
    for i in range(sim_mat.shape[0]):
        sims    = sim_mat[i]
        max_sim = sims.max().item()
        # FIX-6: skip low-confidence positives (mirrors dssm.py max_score > 2 filter)
        if max_sim <= threshold:
            skipped += 1
            continue
        pos_idx = sims.argmax().item()
        neg_idx = sims.argmin().item()
        triplets.append((i, pos_idx, neg_idx))

    print(f"            |  {len(triplets)} triplets  |  skipped: {skipped} "
          f"(max_sim <= {threshold})")
    return triplets


# =============================================================================
#  STEP 5  — Dataset  (matches LegalDataset in dssm.py)
#
#  dssm.py:  pos_state = cat(q_emb(768), pos_candidate_emb(768))  → 1536
#            neg_state = cat(q_emb(768), neg_candidate_emb(768))  → 1536
#  FIX-1 + FIX-2: DSSM input is 1536 (NOT 3072)
# =============================================================================
class LegalDataset(Dataset):
    """
    Each item returns:
      pos_state : cat([q_emb, pos_candidate_emb])   (1536)
      neg_state : cat([q_emb, neg_candidate_emb])   (1536)
    Matches pos_state/neg_state construction in dssm.py exactly.
    """
    def __init__(self, triplets, train_h_q, db_h_q):
        self.triplets  = triplets
        self.train_h_q = train_h_q   # (N_train, 768)
        self.db_h_q    = db_h_q      # (N_db,    768)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q_i, pos_i, neg_i = self.triplets[idx]
        q_emb   = self.train_h_q[q_i]      # (768,)
        pos_emb = self.db_h_q[pos_i]       # (768,)
        neg_emb = self.db_h_q[neg_i]       # (768,)
        pos_state = torch.cat([q_emb, pos_emb], dim=0)   # (1536,)
        neg_state = torch.cat([q_emb, neg_emb], dim=0)   # (1536,)
        return q_i, pos_state, neg_state


class TestLegalDataset(Dataset):
    """
    For validation: returns (q_idx, [all_db_states], db_indices)
    Mirrors TestLegalDataset in dssm.py.
    """
    def __init__(self, triplets, train_h_q, db_h_q, n_db):
        self.triplets  = triplets
        self.train_h_q = train_h_q
        self.db_h_q    = db_h_q
        self.n_db      = n_db

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q_i, pos_i, neg_i = self.triplets[idx]
        q_emb = self.train_h_q[q_i]   # (768,)
        return q_i, pos_i, neg_i, q_emb


# =============================================================================
#  STEP 6  — DSSM Model  (EXACT match to dssm.py)
#
#  dssm.py class dssm:
#    l1 = Linear(768*2, 600)   Xavier Uniform
#    l2 = Linear(600,   300)   Xavier Uniform
#    l3 = Linear(300,   128)   Xavier Uniform
#    out= Linear(128,   1)     Xavier Uniform
#    forward: F.tanh(l1) → F.tanh(l2) → F.tanh(l3) → out.squeeze(-1)
# =============================================================================
class DSSMModel(nn.Module):
    """
    Exact architecture from dssm.py.
    Input  : cat([q_emb(768), candidate_emb(768)])  →  1536-d
    Layers : 1536 → 600 → 300 → 128 → 1
    Init   : Xavier Uniform (all 4 layers)
    Act    : F.tanh (hidden layers), linear output
    """
    def __init__(self):
        super(DSSMModel, self).__init__()
        self.l1  = nn.Linear(768 * 2, 600)   # FIX-1+2: 1536→600
        self.l2  = nn.Linear(600, 300)
        self.l3  = nn.Linear(300, 128)
        self.out = nn.Linear(128, 1)

        # FIX-3: Xavier Uniform (matches dssm.py exactly)
        for layer in [self.l1, self.l2, self.l3, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """x: (batch, 1536)  →  (batch,) scalar score"""
        x1    = F.tanh(self.l1(x))
        x2    = F.tanh(self.l2(x1))
        x3    = F.tanh(self.l3(x2))
        x_out = self.out(x3).squeeze(-1)   # (batch,) — matches dssm.py
        return x_out


# =============================================================================
#  STEP 7  — Training  (exact loop structure from dssm.py)
# =============================================================================
def train(config):
    set_seed(config["seed"])
    device = torch.device(config["device"])
    print(f"\nDevice : {device}\n{'='*66}")

    if config.get("force_fresh"):
        for f in [config["save_path"], config["log_path"]]:
            if os.path.exists(f):
                os.remove(f); print(f"Fresh start |  Deleted {f}")

    _, node_set = load_kg_nodes(config["kg_path"])
    train_data  = load_train_data(config["train_data_path"], node_set)
    db_data     = load_db_data(config["db_data_path"], node_set)

    tok, bert   = load_bert(config["bert_model"], device)

    train_embs  = build_embeddings(train_data, tok, bert, device,
                                   config["train_emb_cache"], "train")
    db_embs     = build_embeddings(db_data,    tok, bert, device,
                                   config["db_emb_cache"], "database")
    del bert
    if device.type == "cuda": torch.cuda.empty_cache()

    # Build triplets with quality filter (FIX-6)
    all_triplets = build_triplets(train_embs, db_embs, config)

    # Train / val split
    random.shuffle(all_triplets)
    n_val        = max(1, int(len(all_triplets) * config["val_split"]))
    val_trips    = all_triplets[:n_val]
    train_trips  = all_triplets[n_val:]

    # DataLoaders (FIX-5: match dssm.py's DataLoader structure)
    train_h_q = train_embs["h_q"]
    db_h_q    = db_embs["h_q"]

    train_ds     = LegalDataset(train_trips, train_h_q, db_h_q)
    val_ds       = TestLegalDataset(val_trips, train_h_q, db_h_q, len(db_data))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False)

    # Model + optimiser (exact dssm.py values)
    dssm      = DSSMModel().to(device)
    optimizer = optim.Adam(dssm.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])
    # FIX-4: use nn.MarginRankingLoss(margin=1) — exact match to dssm.py
    criterion = nn.MarginRankingLoss(margin=config["margin"])

    print(f"\nDSSM        |  Params: {sum(p.numel() for p in dssm.parameters()):,}"
          f"  |  Input: {768*2}  Layers: 1536→600→300→128→1")
    print(f"Training    |  Epochs: {config['epochs']}"
          f"  Batch: {config['batch_size']}"
          f"  Margin: {config['margin']}")
    print(f"Loss        |  nn.MarginRankingLoss(margin={config['margin']})"
          f"  |  Train: {len(train_trips)}  Val: {len(val_trips)}")
    print(f"{'='*66}\n")

    best_val_acc  = 0.0
    log_rows      = []

    for epoch in range(config["epochs"]):
        dssm.train()
        epoch_loss = 0.0

        for batch_keys, pos_states, neg_states in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:02d}/{config['epochs']}",
                leave=False):

            pos_states = pos_states.to(device)   # (B, 1536)
            neg_states = neg_states.to(device)   # (B, 1536)

            pos_score = dssm(pos_states)   # (B,)
            neg_score = dssm(neg_states)   # (B,)

            # FIX-4: MarginRankingLoss with target=+1
            # Equivalent to: max(0, -(pos-neg) + margin) = max(0, neg-pos+margin)
            target = torch.ones(pos_score.shape[0], device=device)
            loss   = criterion(pos_score, neg_score, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dssm.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validation ──────────────────────────────────────────────────────
        dssm.eval()
        val_correct = 0
        val_loss    = 0.0

        with torch.no_grad():
            for q_idxs, pos_idxs, neg_idxs, q_embs in val_loader:
                q_embs   = q_embs.to(device)       # (B, 768)
                pos_embs = db_h_q[pos_idxs].to(device)   # (B, 768)
                neg_embs = db_h_q[neg_idxs].to(device)   # (B, 768)

                pos_st = torch.cat([q_embs, pos_embs], dim=1)   # (B, 1536)
                neg_st = torch.cat([q_embs, neg_embs], dim=1)   # (B, 1536)

                sp = dssm(pos_st)
                sn = dssm(neg_st)
                target = torch.ones(sp.shape[0], device=device)
                val_loss    += criterion(sp, sn, target).item()
                val_correct += (sp > sn).sum().item()

        avg_train_loss = epoch_loss  / max(len(train_loader), 1)
        avg_val_loss   = val_loss    / max(len(val_loader),   1)
        val_acc        = val_correct / max(len(val_trips),    1)

        print(f"Epoch {epoch+1:02d}/{config['epochs']}  |  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  |  "
              f"Val Acc (pos>neg): {val_acc*100:.1f}%")

        log_rows.append({
            "epoch"      : epoch + 1,
            "train_loss" : round(avg_train_loss, 4),
            "val_loss"   : round(avg_val_loss,   4),
            "val_acc"    : round(val_acc,         4),
        })
        pd.DataFrame(log_rows).to_csv(config["log_path"], index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"        : epoch + 1,
                "model_state"  : dssm.state_dict(),
                "optim_state"  : optimizer.state_dict(),
                "config"       : config,
                "best_val_acc" : best_val_acc,
                "best_val_loss": avg_val_loss,
            }, config["save_path"])
            print(f"  ✓ Best model saved  "
                  f"(val_acc={best_val_acc*100:.1f}%  loss={avg_val_loss:.4f})")

        dssm.train()

    print(f"\nTraining complete.")
    print(f"Log   → {config['log_path']}")
    print(f"Model → {config['save_path']}")
    return dssm, db_h_q, db_data


# =============================================================================
#  RETRIEVAL INDEX + INFERENCE
# =============================================================================
def build_retrieval_index(db_h_q, db_data, config):
    torch.save({"h_q": db_h_q.cpu(), "db_data": db_data}, config["retrieval_index"])
    print(f"Index       |  Saved {len(db_data)} DB vectors → {config['retrieval_index']}")


def load_dssm_model(save_path, device):
    ckpt = torch.load(save_path, map_location=device)
    dssm = DSSMModel().to(device)
    dssm.load_state_dict(ckpt["model_state"])
    dssm.eval()
    print(f"Model loaded  |  Best val_acc: {ckpt['best_val_acc']*100:.1f}%"
          f"  (epoch {ckpt['epoch']})")
    return dssm, ckpt["config"]


def retrieve_top_k(query_text, query_chain_q,
                   dssm, tok, bert,
                   db_h_q, db_data,
                   config, device, k=None):
    """
    Retrieve top-K most relevant DB candidates for a given query.
    Input feature = cat([q_emb(768), candidate_emb(768)]) = 1536 — matches DSSMModel.
    """
    if k is None: k = config["top_k"]

    # Encode query question text
    chain_text = " ".join(query_chain_q) if query_chain_q else query_text
    h_q = bert_encode_batch([query_text],  tok, bert, device)   # (1, 768)

    M      = len(db_data)
    scores = []
    dssm.eval()

    with torch.no_grad():
        BATCH = 64
        for start in range(0, M, BATCH):
            end     = min(start + BATCH, M)
            db_embs = db_h_q[start:end].to(device)           # (B, 768)
            q_rep   = h_q.expand(end - start, -1).to(device) # (B, 768)
            state   = torch.cat([q_rep, db_embs], dim=1)     # (B, 1536)
            s       = dssm(state)                             # (B,)
            scores.append(s.cpu())

    scores        = torch.cat(scores)                         # (M,)
    topk_vals, topk_idxs = scores.topk(k)
    results = []
    for score, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        entry         = db_data[idx].copy()
        entry["score"] = round(score, 4)
        results.append(entry)
    return results


def format_rag_context(top_k_results):
    """Paper Eq.11: context_i = [qD_j1,aD_j1, qD_j2,aD_j2, ..., qD_jK,aD_jK]"""
    lines = []
    for i, r in enumerate(top_k_results, 1):
        lines += [f"[Case {i}]",
                  f"Question: {r['question']}",
                  f"Answer  : {r['answer']}", ""]
    return "\n".join(lines)


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":

    # ── Train ──────────────────────────────────────────────────────────────────
    dssm_model, db_h_q, db_data = train(CONFIG)

    device = torch.device(CONFIG["device"])
    build_retrieval_index(db_h_q, db_data, CONFIG)

    # ── Load best checkpoint + inference ──────────────────────────────────────
    dssm_model, cfg = load_dssm_model(CONFIG["save_path"], device)
    index   = torch.load(CONFIG["retrieval_index"], map_location=device)
    db_h_q  = index["h_q"].to(device)
    db_data = index["db_data"]

    tok, bert = load_bert(CONFIG["bert_model"], device)
    _, node_set = load_kg_nodes(CONFIG["kg_path"])
    train_data  = load_train_data(CONFIG["train_data_path"], node_set)

    print(f"\n{'='*66}")
    print("  RAG DEMO  (first 5 training queries)")
    print(f"{'='*66}")

    for i, sample in enumerate(train_data[:5]):
        results = retrieve_top_k(
            sample["question"], sample["chain_q"],
            dssm_model, tok, bert,
            db_h_q, db_data,
            CONFIG, device, k=CONFIG["top_k"])

        print(f"\n── Sample {i+1} ──────────────────────────────────────────")
        print(f"  Query   : {sample['question'][:90]}")
        print(f"  Chain_q : {sample['chain_q']}")
        print(f"\n  Top-{CONFIG['top_k']} Retrieved Cases:")
        for j, r in enumerate(results, 1):
            print(f"    [{j}] Score: {r['score']:+.4f}")
            print(f"        Q: {r['question'][:80]}")
            print(f"        A: {r['answer'][:80]}")
        print(f"\n  ── LLM Context (Eq.11) ──")
        print("  " + format_rag_context(results).replace("\n", "\n  "))
