# =============================================================================
#  train_rl.py  —  LSIM Component 1: Learnable Fact-Rule Chain (REINFORCE)
#  Paper  : "Elevating Legal LLM Responses" (NAACL 2025)
#  Domain : Indian Cyber Law
#
#  Architecture 100% aligned with original policy_network.py (GitHub):
#    ✓  PolicyNetwork  : 768 → 516 → 256 → 128 → N  (ReLU + Kaiming + Softmax)
#    ✓  State          : Fixed BERT CLS embedding of question (never updated)
#    ✓  REINFORCE loss : -sum(log_probs * rewards) — no discount, no normalise
#    ✓  Batch size     : 256   |  Train steps: 5   |  Test steps: 3
#    ✓  Optimizer      : Adam  lr=1e-4,  weight_decay=1e-3
#
#  All bug fixes:
#    FIX-1  get_valid_successors()  — 3-tier fallback; prevents Predicted: []
#    FIX-2  question_embeddings.pt  — separate cache; no node_embeddings.pt clash
#    FIX-3  Train reward tracked    — logged to rl_training_log.csv each epoch
#    FIX-4  Checkpoint resume       — continues training from last saved epoch
#    FIX-5  Fuzzy semantic reward   — substring + section-number matching;
#           'Section 457 of CrPC' ≈ 'Section 457 of The Code Of Criminal Procedure'
#    FIX-6  Epochs raised to 60     — model was still improving at epoch 30
#    FIX-7  FORCE_FRESH flag        — delete stale checkpoint before retraining
# =============================================================================

import os
import json
import copy
import random
import re as _re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from torch.distributions import Categorical
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
#  CONFIG
# =============================================================================
CONFIG = {
    # ── paths ─────────────────────────────────────────────────────────────────
    "data_path"      : "ground_truth_rl_dataset.csv",
    "kg_path"        : "fact_rule_chains.py",
    "emb_cache_path" : "question_embeddings.pt",
    "save_path"      : "rl_policy_model.pt",
    "log_path"       : "rl_training_log.csv",

    # ── bert ──────────────────────────────────────────────────────────────────
    "bert_model"     : "bert-base-uncased",

    # ── training (paper values) ───────────────────────────────────────────────
    "epochs"         : 100,       # FIX-6: raised from 30 (model still improving)
    "batch_size"     : 256,
    "lr"             : 1e-4,
    "weight_decay"   : 1e-3,
    "train_steps"    : 5,
    "test_steps"     : 3,
    "grad_clip"      : 1.0,

    # ── FIX-5: fuzzy reward ───────────────────────────────────────────────────
    # Partial reward for near-matches:
    #   exact   match  → 1.0
    #   substring      → 0.7  (e.g. 'Section 457 of CrPC' in 'Section 457 of ...')
    #   section-number → 0.5  (same section/article numbers in both strings)
    "use_fuzzy_reward"       : True,
    "fuzzy_substring_score"  : 0.7,
    "fuzzy_number_score"     : 0.5,

    # ── FIX-7: fresh start flag ───────────────────────────────────────────────
    # Set True to delete stale checkpoint and retrain from epoch 1.
    # IMPORTANT: set True on first run after this update, then back to False.
    "force_fresh"    : True,    #True

    # ── misc ──────────────────────────────────────────────────────────────────
    "seed"           : 42,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    "val_split"      : 0.1,
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
#  STEP 1 — Build NetworkX DiGraph from fact_rule_chains.py
# =============================================================================
def load_kg(kg_path):
    with open(kg_path, "r", encoding="utf-8") as f:
        src = f.read()
    ns = {}
    exec(src, ns)

    facts = ns["facts"]
    rules = ns["rules"]
    edges = ns["edges"]

    G = nx.DiGraph()
    for n in facts: G.add_node(n, node_type="fact")
    for n in rules: G.add_node(n, node_type="rule")
    for src_n, dst_n, rel in edges:
        G.add_edge(src_n, dst_n, relation=rel)

    all_nodes = list(G.nodes())
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    idx2node  = {i: n for i, n in enumerate(all_nodes)}

    leaf_count = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
    print(f"KG loaded   |  Nodes: {G.number_of_nodes()}"
          f"  Facts: {len(facts)}  Rules: {len(rules)}"
          f"  Edges: {G.number_of_edges()}"
          f"  Leaf nodes: {leaf_count} ({leaf_count/G.number_of_nodes()*100:.1f}%)")
    return G, all_nodes, node2idx, idx2node


# =============================================================================
#  STEP 2 — Load dataset (stable idx assigned BEFORE shuffling)
# =============================================================================
def load_data(data_path, node2idx):
    df      = pd.read_csv(data_path)
    data    = []
    skipped = 0
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row["chain_q"]) if n in node2idx]
        ca = [n for n in json.loads(row["chain_a"]) if n in node2idx]
        if not cq or not ca:
            skipped += 1
            continue
        data.append({
            "idx"      : len(data),      # stable CSV-order index
            "question" : str(row["question"]),
            "answer"   : str(row["answer"]),
            "chain_q"  : cq,
            "chain_a"  : ca,
        })
    print(f"Dataset     |  Loaded: {len(data)}  Skipped: {skipped}")
    return data


# =============================================================================
#  STEP 3 — BERT question embeddings (pre-computed, cached)
# =============================================================================
def precompute_embeddings(data, model_name, cache_path, device):
    if os.path.exists(cache_path):
        print(f"Embeddings  |  Loading cache ← {cache_path}")
        emb_dict = torch.load(cache_path, map_location=device)
        print(f"            |  {len(emb_dict)} question embeddings  dim=768")
        return emb_dict

    print(f"Embeddings  |  Pre-computing BERT for {len(data)} questions "
          f"(runs once, then cached) ...")
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()

    emb_dict = {}
    BATCH    = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(data), BATCH), desc="  BERT encode"):
            batch = data[i : i + BATCH]
            texts = [d["question"] for d in batch]
            enc   = tok(texts, return_tensors="pt", truncation=True,
                        max_length=128, padding=True).to(device)
            cls   = bert(**enc).last_hidden_state[:, 0, :]
            for j, emb in enumerate(cls):
                emb_dict[batch[j]["idx"]] = emb.cpu()

    torch.save(emb_dict, cache_path)
    print(f"            |  Saved → {cache_path}")
    return torch.load(cache_path, map_location=device)


# =============================================================================
#  POLICY NETWORK  ← exact paper architecture
#  768 → 516 → 256 → 128 → output_size | ReLU + Kaiming Normal + Softmax
# =============================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, output_size):
        super(PolicyNetwork, self).__init__()
        self.l1  = nn.Linear(768, 516)
        self.l2  = nn.Linear(516, 256)
        self.l3  = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_size)

        nn.init.kaiming_normal_(self.l1.weight,  mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.l2.weight,  mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.l3.weight,  mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.out.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return F.softmax(self.out(x), dim=1)


# =============================================================================
#  FIX-1 — get_valid_successors()  (3-tier fallback)
#
#  31.8% of KG nodes are leaf nodes (no successors); 99.7% of rule nodes.
#  Tier 1 → successors of LAST node         (paper's original logic)
#  Tier 2 → successors of ANY chain node    (when last node is a leaf)
#  Tier 3 → predecessors of all chain nodes (last resort)
# =============================================================================
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


# =============================================================================
#  FIX-5 — Fuzzy semantic reward
#
#  Motivation: KG has multiple string representations of the same legal entity.
#  Example:
#    'Section 457 of CrPC'  ≈  'Section 457 of The Code Of Criminal Procedure'
#  Without fuzzy reward the policy gets reward=0 even when it finds the right law.
#
#  Scoring:
#    exact match        → 1.0
#    substring match    → fuzzy_substring_score (0.7)
#    section-number match → fuzzy_number_score  (0.5)
#    no match           → 0.0
# =============================================================================
def compute_reward(node, golden_ca_set, config):
    """
    Compute reward for selecting `node` given golden chain_a set.
    Uses exact → substring → section-number matching hierarchy.
    """
    # Exact match
    if node in golden_ca_set:
        return 1.0

    if not config["use_fuzzy_reward"]:
        return 0.0

    node_norm = node.lower().strip()

    for g in golden_ca_set:
        g_norm = g.lower().strip()

        # Substring match (either direction)
        if node_norm in g_norm or g_norm in node_norm:
            return config["fuzzy_substring_score"]

        # Section / article number match
        # Extract all digit sequences e.g. "457", "66", "383"
        node_nums = set(_re.findall(r'\b\d+\b', node_norm))
        gold_nums = set(_re.findall(r'\b\d+\b', g_norm))
        if node_nums and gold_nums and node_nums & gold_nums:
            return config["fuzzy_number_score"]

    return 0.0


# =============================================================================
#  SELECT ACTION  ← exact paper logic
# =============================================================================
def select_action(policy_net, q_embedding, successors, node2idx, num_nodes, device):
    if not successors:
        return None, None

    probabilities = policy_net(q_embedding).squeeze()

    valid_idx = [node2idx[n] for n in successors if n in node2idx]
    if not valid_idx:
        return None, None

    mask = torch.zeros(num_nodes, device=device)
    mask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = 1.0

    filtered = probabilities * mask
    total    = filtered.sum()
    if total.item() == 0:
        return None, None
    filtered = filtered / total

    dist     = Categorical(filtered)
    action   = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


# =============================================================================
#  TRAINING
# =============================================================================
def train(config):
    set_seed(config["seed"])
    device = torch.device(config["device"])
    print(f"\nDevice : {device}\n{'='*66}")

    # FIX-7: delete stale checkpoint if force_fresh is set
    if config.get("force_fresh") and os.path.exists(config["save_path"]):
        os.remove(config["save_path"])
        print(f"FIX-7  |  Deleted stale checkpoint '{config['save_path']}'"
              f" → training from scratch")
    if config.get("force_fresh") and os.path.exists(config["log_path"]):
        os.remove(config["log_path"])

    G, all_nodes, node2idx, idx2node = load_kg(config["kg_path"])
    all_data = load_data(config["data_path"], node2idx)   # stable idx here

    emb_dict = precompute_embeddings(
        all_data, config["bert_model"], config["emb_cache_path"], device)

    random.shuffle(all_data)
    n_val      = max(1, int(len(all_data) * config["val_split"]))
    val_data   = all_data[:n_val]
    train_data = all_data[n_val:]
    print(f"Split       |  Train: {len(train_data)}  Val: {len(val_data)}")

    num_nodes  = len(all_nodes)
    policy_net = PolicyNetwork(output_size=num_nodes).to(device)
    optimizer  = optim.Adam(policy_net.parameters(),
                            lr=config["lr"],
                            weight_decay=config["weight_decay"])

    batch_size = config["batch_size"]
    n_batches  = (len(train_data) + batch_size - 1) // batch_size

    # FIX-4: resume from checkpoint (only if force_fresh is False)
    start_epoch = 0
    log_rows    = []
    best_val    = -float("inf")

    if not config.get("force_fresh") and os.path.exists(config["save_path"]):
        print(f"\nCheckpoint  |  Found '{config['save_path']}' — resuming ...")
        ckpt = torch.load(config["save_path"], map_location=device)
        policy_net.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val_reward"]
        if os.path.exists(config["log_path"]):
            log_rows = pd.read_csv(config["log_path"]).to_dict("records")
        print(f"            |  Resuming from epoch {start_epoch + 1}"
              f"  (best_val = {best_val:.4f})")
    else:
        print(f"\nCheckpoint  |  Training from scratch (epoch 1)")

    print(f"\nPolicy      |  Params: {sum(p.numel() for p in policy_net.parameters()):,}"
          f"  Output: {num_nodes} nodes")
    print(f"Training    |  Epochs: {config['epochs']}  "
          f"Batch: {batch_size}  Steps/sample: {config['train_steps']}")
    print(f"Reward      |  Fuzzy matching: {config['use_fuzzy_reward']}"
          f"  (substr={config['fuzzy_substring_score']}"
          f"  number={config['fuzzy_number_score']})")
    print(f"{'='*66}\n")

    for epoch in range(start_epoch, config["epochs"]):
        random.shuffle(train_data)
        policy_net.train()
        epoch_loss         = 0.0
        epoch_train_reward = 0.0

        for b_j in tqdm(range(n_batches),
                        desc=f"Epoch {epoch+1:02d}/{config['epochs']}",
                        leave=False):

            start      = b_j * batch_size
            end        = min(start + batch_size, len(train_data))
            batch      = train_data[start:end]
            batch_loss = torch.zeros(1, device=device, requires_grad=True)

            for sample in batch:
                q_cot   = copy.deepcopy(sample["chain_q"])
                a_cot   = set(sample["chain_a"])
                q_emb   = emb_dict[sample["idx"]].unsqueeze(0).to(device)
                visited = set(q_cot)

                log_prob_5 = []
                reward_5   = []

                for _ in range(config["train_steps"]):
                    successors = get_valid_successors(G, q_cot, visited)
                    if not successors:
                        break

                    action, log_prob = select_action(
                        policy_net, q_emb, successors,
                        node2idx, num_nodes, device)
                    if action is None:
                        break

                    next_node = idx2node[action.item()]
                    # FIX-5: fuzzy semantic reward
                    reward    = compute_reward(next_node, a_cot, config)

                    q_cot.append(next_node)
                    visited.add(next_node)
                    log_prob_5.append(log_prob)
                    reward_5.append(reward)

                if not log_prob_5:
                    continue

                log_probs_t = torch.stack(log_prob_5)
                rewards_t   = torch.tensor(reward_5, dtype=torch.float32,
                                           device=device)
                sample_loss = -torch.sum(log_probs_t * rewards_t)
                batch_loss  = batch_loss + sample_loss
                epoch_train_reward += sum(reward_5)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_net.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_loss += batch_loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        policy_net.eval()
        val_total_reward = 0.0

        with torch.no_grad():
            for sample in val_data:
                q_cot   = copy.deepcopy(sample["chain_q"])
                a_cot   = set(sample["chain_a"])
                q_emb   = emb_dict[sample["idx"]].unsqueeze(0).to(device)
                visited = set(q_cot)

                for _ in range(config["test_steps"]):
                    successors = get_valid_successors(G, q_cot, visited)
                    if not successors:
                        break
                    action, _ = select_action(
                        policy_net, q_emb, successors,
                        node2idx, num_nodes, device)
                    if action is None:
                        break
                    next_node = idx2node[action.item()]
                    # FIX-5: fuzzy reward in validation too
                    val_total_reward += compute_reward(next_node, a_cot, config)
                    q_cot.append(next_node)
                    visited.add(next_node)

        avg_loss         = epoch_loss         / max(n_batches, 1)
        avg_train_reward = epoch_train_reward / max(len(train_data), 1)
        avg_val_reward   = val_total_reward   / max(len(val_data), 1)

        print(f"Epoch {epoch+1:02d}/{config['epochs']}  |  "
              f"Loss: {avg_loss:+.4f}  |  "
              f"Train Reward: {avg_train_reward:.4f}  |  "
              f"Val Reward: {avg_val_reward:.4f}")

        log_rows.append({
            "epoch"        : epoch + 1,
            "train_loss"   : round(avg_loss,         4),
            "train_reward" : round(avg_train_reward, 4),
            "val_reward"   : round(avg_val_reward,   4),
        })
        pd.DataFrame(log_rows).to_csv(config["log_path"], index=False)

        if avg_val_reward > best_val:
            best_val = avg_val_reward
            torch.save({
                "epoch"               : epoch + 1,
                "policy_state_dict"   : policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config"              : config,
                "all_nodes"           : all_nodes,
                "node2idx"            : node2idx,
                "idx2node"            : idx2node,
                "best_val_reward"     : best_val,
            }, config["save_path"])
            print(f"  ✓ Best model saved  (val_reward = {best_val:.4f})")

        policy_net.train()

    print(f"\nTraining complete.")
    print(f"Log   → {config['log_path']}")
    print(f"Model → {config['save_path']}")
    return policy_net, G, all_nodes, node2idx, idx2node, emb_dict


# =============================================================================
#  INFERENCE
# =============================================================================
def load_model(save_path, device):
    ckpt       = torch.load(save_path, map_location=device)
    all_nodes  = ckpt["all_nodes"]
    node2idx   = ckpt["node2idx"]
    idx2node   = ckpt["idx2node"]
    cfg        = ckpt["config"]

    policy_net = PolicyNetwork(output_size=len(all_nodes)).to(device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    policy_net.eval()

    print(f"Model loaded  |  Best val_reward: {ckpt['best_val_reward']:.4f}"
          f"  (epoch {ckpt['epoch']})")
    return policy_net, all_nodes, node2idx, idx2node, cfg


def predict_chain_a(chain_q, q_embedding,
                    policy_net, G, all_nodes, node2idx, idx2node,
                    device, config):
    policy_net.eval()
    q_cot     = copy.deepcopy(chain_q)
    visited   = set(q_cot)
    predicted = []

    with torch.no_grad():
        for _ in range(config["test_steps"]):
            successors = get_valid_successors(G, q_cot, visited)
            if not successors:
                break
            action, _ = select_action(
                policy_net, q_embedding, successors,
                node2idx, len(all_nodes), device)
            if action is None:
                break
            node = idx2node[action.item()]
            predicted.append(node)
            q_cot.append(node)
            visited.add(node)

    return predicted


def evaluate_hit(pred_list, golden_list, config):
    """
    Compute exact and fuzzy hits for an inference sample.
    Returns (exact_hits, fuzzy_hits, total_golden).
    """
    golden_set  = set(golden_list)
    exact_hits  = sum(1 for p in pred_list if p in golden_set)
    fuzzy_hits  = sum(1 for p in pred_list
                      if p not in golden_set
                      and compute_reward(p, golden_set, config) > 0)
    return exact_hits, fuzzy_hits, len(golden_list)


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":

    # ── Train ──────────────────────────────────────────────────────────────────
    (policy_net, G,
     all_nodes, node2idx, idx2node,
     emb_dict) = train(CONFIG)

    # ── Inference Demo ─────────────────────────────────────────────────────────
    device   = torch.device(CONFIG["device"])
    all_data = load_data(CONFIG["data_path"], node2idx)

    print(f"\n{'='*66}")
    print("  INFERENCE DEMO  (first 5 samples)")
    print(f"{'='*66}")

    total_exact = total_fuzzy = total_gold = 0
    for i, sample in enumerate(all_data[:5]):
        q_emb = emb_dict[sample["idx"]].unsqueeze(0).to(device)
        pred  = predict_chain_a(
            sample["chain_q"], q_emb,
            policy_net, G, all_nodes, node2idx, idx2node,
            device, CONFIG)
        e, f, g = evaluate_hit(pred, sample["chain_a"], CONFIG)
        total_exact += e; total_fuzzy += f; total_gold += g

        print(f"\nSample {i+1}")
        print(f"  chain_q   : {sample['chain_q']}")
        print(f"  Predicted : {pred}")
        print(f"  Golden    : {sample['chain_a']}")
        print(f"  Exact: {e}/{g}  |  Fuzzy (near-match): {f}/{g}")

    print(f"\n  ── Overall (5 samples) ──────────────────────────────")
    print(f"  Exact hits : {total_exact} / {total_gold}"
          f"  ({total_exact/total_gold*100:.1f}%)")
    print(f"  +Fuzzy hits: {total_exact+total_fuzzy} / {total_gold}"
          f"  ({(total_exact+total_fuzzy)/total_gold*100:.1f}%)")
