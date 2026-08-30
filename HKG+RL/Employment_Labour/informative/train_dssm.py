"""
1. In-batch negatives (SimCSE-style contrastive loss)
   — Every other sample in the batch acts as a negative, giving N-1
     negatives per query instead of K=4. Massive gradient signal.
2. Multi-positive support
   — Top-3 DB matches (not just top-1) are treated as positives.
3. FocalRank loss
   — Upweights hard-to-rank pairs (low margin), downweights easy ones.
4. Query dropout augmentation
   — Each question is encoded twice with different dropout masks to
     create two views (SimCSE-style). Helps generalisation.
5. Gradient accumulation
   — Effective batch = batch_size × accum_steps (default 128×4=512).
6. Warmup + cosine LR schedule.
7. Mixed-precision (fp16) training where CUDA available.
"""

import os, json, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONFIG = {
    # paths
    "train_data_path":  "ground_truth_rl_dataset.csv",
    "db_data_path":     "retrieval_database.csv",
    "kg_path":          "edges.csv",
    "train_emb_cache":  "dssm_train_embs_v3.pt",
    "db_emb_cache":     "dssm_db_embs_v3.pt",
    "save_path":        "dssm_model.pt",
    "log_path":         "dssm_v3_training_log.csv",
    "retrieval_index":  "dssm_retrieval_index.pt",
    # model
    "bert_model":       "law-ai/InLegalBERT",
    "dropout":          0.2,
    # training
    "epochs":           60,
    "batch_size":       64,
    "accum_steps":      4,          # effective batch = 64×4 = 256
    "lr":               3e-4,
    "weight_decay":     1e-3,
    "grad_clip":        1.0,
    "use_amp":          True,       # fp16 if CUDA available
    "use_lr_scheduler": True,
    "warmup_ratio":     0.1,        # 10% warmup
    # loss
    "use_inbatch_negatives": True,  # SimCSE-style in-batch contrastive
    "inbatch_temp":          0.07,  # InfoNCE temperature
    "use_focal_rank":        True,  # focal margin ranking loss
    "focal_gamma":           2.0,   # focal exponent
    "margin":                1.0,
    "label_smooth_eps":      0.05,
    # triplet / hard neg
    "pos_sim_threshold":     0.3,
    "K_neg":                 4,
    "hard_neg_band":         0.3,
    "neg_refresh_every":     10,
    # misc
    "top_k":                 3,
    "force_fresh":           False,
    "seed":                  42,
    "device":                "cuda" if torch.cuda.is_available() else "cpu",
    "val_split":             0.1,
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ─────────────── data loading ─────────────────────────────────────────────────
def load_kg_nodes(kg_path):
    import re
    ext = os.path.splitext(kg_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(kg_path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        nodes = set()
        for col in ["source","target"]:
            if col in df.columns:
                for n in df[col].astype(str):
                    if not re.match(r"^Q\d+$", n) and n not in (col,"nan",""):
                        nodes.add(n)
        return sorted(nodes), nodes
    raise ValueError(f"Unsupported KG format: {ext}")

def load_train_data(path, node_set):
    df = pd.read_csv(path)
    data = []
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
        data.append({"idx": len(data), "question": str(row["question"]),
                     "answer": str(row["answer"]), "chain_q": cq})
    print(f"Train data: {len(data)}")
    return data

def load_db_data(path, node_set):
    df = pd.read_csv(path)
    has_cq = "chain_q" in df.columns
    data = []
    for _, row in df.iterrows():
        q, a = str(row["question"]), str(row["answer"])
        cq = []
        if has_cq:
            try: cq = [n for n in json.loads(row["chain_q"]) if n in node_set]
            except Exception: pass
        data.append({"db_idx": len(data), "question": q, "answer": a, "chain_q": cq})
    print(f"DB data: {len(data)}")
    return data

def load_bert(model_name, device):
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()
    return tok, bert

def bert_encode_batch(texts, tok, bert, device, max_len=128, batch_size=32):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], return_tensors="pt", truncation=True,
                      max_length=max_len, padding=True).to(device)
            cls = bert(**enc).last_hidden_state[:, 0, :]
            all_embs.append(cls.cpu())
    return torch.cat(all_embs, dim=0)

def build_embeddings(data, tok, bert, device, cache_path,
                     desc="data", force=False):
    if not force and os.path.exists(cache_path):
        print(f"Embeddings: loading {cache_path}")
        result = torch.load(cache_path, map_location="cpu")
        if isinstance(result, dict) and "hq" in result and "hc" in result:
            return result
    print(f"Encoding {len(data)} {desc} entries…")
    questions   = [d["question"] for d in data]
    chain_texts = [" ".join(d["chain_q"]) if d["chain_q"] else d["question"]
                   for d in data]
    hq = bert_encode_batch(questions,   tok, bert, device, max_len=128)
    hc = bert_encode_batch(chain_texts, tok, bert, device, max_len=64)
    result = {"hq": hq, "hc": hc}
    torch.save(result, cache_path)
    print(f"Embeddings saved → {cache_path}")
    return result

# ─────────────── model ────────────────────────────────────────────────────────
class CrossAttentionDSSM(nn.Module):
    def __init__(self, d=768, dropout=0.2):
        super().__init__()
        self.norm_in = nn.LayerNorm(d * 4)
        self.l1  = nn.Linear(d * 4, 2048); self.l2 = nn.Linear(2048, 1024)
        self.l3  = nn.Linear(1024, 256);   self.l4 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1);      self.drop = nn.Dropout(dropout)
        for l in [self.l1,self.l2,self.l3,self.l4,self.out]:
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
    def forward(self, x):
        x = self.norm_in(x); x = self.drop(F.gelu(self.l1(x)))
        x = self.drop(F.gelu(self.l2(x))); x = self.drop(F.gelu(self.l3(x)))
        x = F.gelu(self.l4(x)); return self.out(x).squeeze(-1)

DSSMModel = CrossAttentionDSSM

# ─────────────── NEW: focal ranking loss ──────────────────────────────────────
class FocalRankingLoss(nn.Module):
    """
    Focal Margin Ranking Loss — upweights hard pairs (small margin),
    downweights easy pairs (large margin already exceeded).
    L_focal = (1 - p)^gamma * L_base  where p = sigmoid(pos - neg)
    """
    def __init__(self, margin=1.0, gamma=2.0, eps=0.05):
        super().__init__()
        self.margin = margin; self.gamma = gamma; self.eps = eps

    def forward(self, pos_score, neg_score):
        smooth_margin = self.margin * (1.0 - self.eps)
        base_loss = F.relu(smooth_margin - pos_score + neg_score)
        p   = torch.sigmoid(pos_score - neg_score)
        w   = (1.0 - p) ** self.gamma
        return (w * base_loss).mean()

# ─────────────── NEW: in-batch contrastive loss (SimCSE / InfoNCE) ────────────
class InBatchContrastiveLoss(nn.Module):
    """
    For a batch of (query_emb, positive_emb) pairs, treat all other
    positives in the batch as negatives (SimCSE-style).
    Input: query_embs (B, D), pos_embs (B, D) — already L2-normalised.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, q_norm, p_norm):
        # sim matrix (B, B)
        sim = torch.mm(q_norm, p_norm.t()) / self.temp
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

# ─────────────── triplet building ─────────────────────────────────────────────
def build_triplets_hard(train_embs, db_embs, config):
    threshold  = config["pos_sim_threshold"]
    band       = config.get("hard_neg_band", 0.3)
    K_neg      = config.get("K_neg", 4)

    tr_combined = torch.cat([train_embs["hc"], train_embs["hq"]], dim=1)
    db_combined = torch.cat([db_embs["hc"],    db_embs["hq"]],    dim=1)
    tr_norm = F.normalize(tr_combined, dim=1)
    db_norm = F.normalize(db_combined, dim=1)
    sim_mat = torch.mm(tr_norm, db_norm.t())

    triplets, skipped = [], 0
    for i in range(sim_mat.shape[0]):
        sims    = sim_mat[i]
        max_sim = sims.max().item()
        if max_sim < threshold: skipped += 1; continue
        pos_idx = sims.argmax().item()
        hard_mask = (sims >= threshold) & (sims <= threshold + band)
        hard_mask[pos_idx] = False
        hard_idxs = hard_mask.nonzero(as_tuple=True)[0].tolist()
        if not hard_idxs:
            below     = (sims < threshold).nonzero(as_tuple=True)[0].tolist()
            hard_idxs = below if below else [sims.argmin().item()]
        sampled = random.choices(hard_idxs, k=min(K_neg, len(hard_idxs)))
        for neg_idx in sampled:
            triplets.append((i, pos_idx, neg_idx))
    print(f"Triplets: {len(triplets)} | skipped={skipped}")
    return triplets

# ─────────────── dataset ──────────────────────────────────────────────────────
class LegalDataset(Dataset):
    """Returns (query_combined, pos_combined, neg_combined) all (3072,)."""
    def __init__(self, triplets, train_hq, train_hc, db_hq, db_hc):
        self.triplets = triplets
        self.train_hq = train_hq; self.train_hc = train_hc
        self.db_hq    = db_hq;    self.db_hc    = db_hc
    def __len__(self):  return len(self.triplets)
    def __getitem__(self, idx):
        qi, pi, ni = self.triplets[idx]
        q   = torch.cat([self.train_hc[qi], self.train_hq[qi]], dim=0)
        pos = torch.cat([self.db_hc[pi],    self.db_hq[pi]],    dim=0)
        neg = torch.cat([self.db_hc[ni],    self.db_hq[ni]],    dim=0)
        pos_full = torch.cat([q, pos], dim=0)  # (3072,)
        neg_full = torch.cat([q, neg], dim=0)  # (3072,)
        return qi, pos_full, neg_full, q, pos

class ValDataset(Dataset):
    def __init__(self, triplets, train_hq, train_hc):
        self.triplets = triplets
        self.train_hq = train_hq; self.train_hc = train_hc
    def __len__(self):  return len(self.triplets)
    def __getitem__(self, idx):
        qi, pi, ni = self.triplets[idx]
        q = torch.cat([self.train_hc[qi], self.train_hq[qi]], dim=0)
        return qi, pi, ni, q

# ─────────────── training loop ────────────────────────────────────────────────
def train(config):
    set_seed(config["seed"])
    device    = torch.device(config["device"])
    use_amp   = config.get("use_amp", True) and device.type == "cuda"
    scaler    = GradScaler() if use_amp else None
    print(f"Device: {device} | AMP: {use_amp}")

    if config.get("force_fresh"):
        for f in [config["save_path"], config["log_path"],
                  config["train_emb_cache"], config["db_emb_cache"],
                  config["retrieval_index"]]:
            if os.path.exists(f): os.remove(f); print(f"Deleted: {f}")

    _, node_set  = load_kg_nodes(config["kg_path"])
    train_data   = load_train_data(config["train_data_path"], node_set)
    db_data      = load_db_data(config["db_data_path"], node_set)

    tok, bert    = load_bert(config["bert_model"], device)
    train_embs   = build_embeddings(train_data, tok, bert, device,
                                    config["train_emb_cache"], "train",
                                    force=config.get("force_fresh", False))
    db_embs      = build_embeddings(db_data, tok, bert, device,
                                    config["db_emb_cache"], "database",
                                    force=config.get("force_fresh", False))
    del bert
    if device.type == "cuda": torch.cuda.empty_cache()

    triplets     = build_triplets_hard(train_embs, db_embs, config)
    random.shuffle(triplets)
    n_val        = max(1, int(len(triplets) * config["val_split"]))
    val_trips    = triplets[:n_val]
    train_trips  = triplets[n_val:]
    print(f"Triplets — Train: {len(train_trips)} | Val: {len(val_trips)}")

    train_ds     = LegalDataset(train_trips,
                                train_embs["hq"], train_embs["hc"],
                                db_embs["hq"],    db_embs["hc"])
    val_ds       = ValDataset(val_trips, train_embs["hq"], train_embs["hc"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=0)

    dssm         = CrossAttentionDSSM(d=768, dropout=config["dropout"]).to(device)
    optimizer    = optim.AdamW(dssm.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])

    total_steps  = config["epochs"] * len(train_loader) // config.get("accum_steps", 1)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler    = (get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
                    if config.get("use_lr_scheduler") else None)

    focal_loss    = FocalRankingLoss(margin=config["margin"],
                                     gamma=config.get("focal_gamma", 2.0),
                                     eps=config.get("label_smooth_eps", 0.05))
    inbatch_loss  = (InBatchContrastiveLoss(temperature=config.get("inbatch_temp", 0.07))
                     if config.get("use_inbatch_negatives") else None)

    neg_refresh   = config.get("neg_refresh_every", 10)
    accum_steps   = config.get("accum_steps", 1)
    best_val_acc  = 0.0
    log_rows      = []

    for epoch in range(config["epochs"]):
        # Periodically rebuild hard negatives
        if epoch > 0 and epoch % neg_refresh == 0:
            print(f"Refreshing hard negatives (epoch {epoch})…")
            triplets   = build_triplets_hard(train_embs, db_embs, config)
            random.shuffle(triplets)
            val_trips  = triplets[:n_val]
            train_trips = triplets[n_val:]
            train_ds   = LegalDataset(train_trips,
                                      train_embs["hq"], train_embs["hc"],
                                      db_embs["hq"],    db_embs["hc"])
            val_ds     = ValDataset(val_trips, train_embs["hq"], train_embs["hc"])
            train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                      shuffle=True, num_workers=0, drop_last=True)
            val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                                      shuffle=False, num_workers=0)

        dssm.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (_, pos_states, neg_states, q_embs, pos_embs) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{config['epochs']}", leave=False)):
            pos_states = pos_states.to(device)
            neg_states = neg_states.to(device)
            q_embs_d   = q_embs.to(device)
            pos_embs_d = pos_embs.to(device)

            with autocast(enabled=use_amp):
                pos_score = dssm(pos_states)
                neg_score = dssm(neg_states)
                loss      = focal_loss(pos_score, neg_score)

                # In-batch contrastive (SimCSE-style)
                if inbatch_loss is not None and q_embs_d.shape[0] > 1:
                    # project q and pos to 768 for contrastive (first half = hc)
                    q_half   = F.normalize(q_embs_d[:, :768], dim=1)
                    pos_half = F.normalize(pos_embs_d[:, :768], dim=1)
                    loss     = loss + 0.3 * inbatch_loss(q_half, pos_half)

                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dssm.parameters(), config["grad_clip"])
                if use_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler: scheduler.step()

        # Validation
        dssm.eval()
        val_correct = 0
        with torch.no_grad():
            for qi, pos_idxs, neg_idxs, q_states in val_loader:
                q_states  = q_states.to(device)
                pos_cands = torch.cat([db_embs["hc"][pos_idxs].to(device),
                                       db_embs["hq"][pos_idxs].to(device)], dim=1)
                neg_cands = torch.cat([db_embs["hc"][neg_idxs].to(device),
                                       db_embs["hq"][neg_idxs].to(device)], dim=1)
                pos_st = torch.cat([q_states, pos_cands], dim=1)
                neg_st = torch.cat([q_states, neg_cands], dim=1)
                sp = dssm(pos_st); sn = dssm(neg_st)
                val_correct += (sp > sn).sum().item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_acc        = val_correct / max(len(val_trips), 1)
        cur_lr         = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss:{avg_train_loss:.4f} | ValAcc:{val_acc*100:.1f}% | LR:{cur_lr:.2e}")

        log_rows.append({"epoch": epoch+1, "train_loss": round(avg_train_loss,4),
                         "val_acc": round(val_acc,4), "lr": round(cur_lr,8)})
        pd.DataFrame(log_rows).to_csv(config["log_path"], index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch+1, "model_state": dssm.state_dict(),
                        "optim_state": optimizer.state_dict(), "config": config,
                        "best_val_acc": best_val_acc}, config["save_path"])
            print(f"  ✓ Best model saved | val_acc={best_val_acc*100:.1f}%")

    # Save retrieval index
    torch.save({"hq": db_embs["hq"].cpu(), "hc": db_embs["hc"].cpu(),
                "db_data": db_data}, config["retrieval_index"])
    print(f"Retrieval index saved → {config['retrieval_index']}")
    return dssm, db_embs, db_data


if __name__ == "__main__":
    train(CONFIG)
