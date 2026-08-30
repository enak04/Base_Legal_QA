"""
Architecture improvements :
1. ChainAttention — soft-attention pooling over chain nodes
2. ResidualBlock  — skip connections in policy MLP
3. PPO clip       — stable policy updates
4. GAE            — better advantage estimation
5. Neighbour shaping reward — credit for bridge nodes
"""

import os, json, copy, random, re
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

CONFIG = {
    # ── paths ────────────────────────────────────────────────────────────────
    'data_path':            'ground_truth_rl_dataset.csv',
    'kg_path':              'edges.csv',
    'emb_cache_path':       'question_embeddings.pt',
    'node_emb_cache_path':  'node_embeddings.pt',
    'save_path':            'rl_policy_model.pt',
    'log_path':             'rl_training_log.csv',
    # ── model ─────────────────────────────────────────────────────────────
    'bert_model':           'law-ai/InLegalBERT',
    # ── training ──────────────────────────────────────────────────────────
    'epochs':               100,
    'batch_size':           256,
    'lr':                   1e-4,
    'weight_decay':         1e-3,
    'train_steps':          5,
    'test_steps':           3,
    'grad_clip':            1.0,
    # ── reward ────────────────────────────────────────────────────────────
    'use_fuzzy_reward':          True,
    'fuzzy_substring_score':     0.7,
    'fuzzy_number_score':        0.5,
    'neighbour_shaping_score':   0.3,
    # ── PPO ───────────────────────────────────────────────────────────────
    'use_ppo':              True,
    'ppo_clip_eps':         0.2,
    'ppo_epochs':           3,          # inner PPO update epochs per batch
    # ── GAE ───────────────────────────────────────────────────────────────
    'use_gae':              True,
    'gamma':                0.95,
    'gae_lambda':           0.95,
    # ── stability ─────────────────────────────────────────────────────────
    'entropy_coef':         0.01,
    'baseline_decay':       0.99,
    'temp_init':            2.0,
    'use_lr_scheduler':     True,
    # ── misc ──────────────────────────────────────────────────────────────
    'force_fresh':          False,
    'seed':                 42,
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'val_split':            0.1,
}

# ─────────────────────── utilities ────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_kg(kg_path):
    ext = os.path.splitext(kg_path)[1].lower()
    if ext == '.pkl':
        import pickle
        with open(kg_path, 'rb') as f: g_raw = pickle.load(f)
        G = nx.DiGraph(g_raw) if not isinstance(g_raw, nx.DiGraph) else g_raw
    elif ext == '.csv':
        G = nx.DiGraph()
        df = pd.read_csv(kg_path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            s = str(row['source']).strip(); t = str(row['target']).strip()
            if s in ('source', 'nan', '') or t in ('target', 'nan', ''): continue
            G.add_node(s); G.add_node(t)
            G.add_edge(s, t, relation=str(row.get('relation', 'RELATED')).strip())
    else:
        raise ValueError(f"Unsupported KG format: {ext}")
    all_nodes = list(G.nodes)
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    idx2node  = {i: n for i, n in enumerate(all_nodes)}
    print(f"KG | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G, all_nodes, node2idx, idx2node

def load_data(path, node2idx):
    df = pd.read_csv(path); data, skip = [], 0
    for _, row in df.iterrows():
        cq = [n for n in json.loads(row['chain_q']) if n in node2idx]
        ca = [n for n in json.loads(row['chain_a']) if n in node2idx]
        if not cq or not ca: skip += 1; continue
        data.append({'idx': len(data), 'question': str(row['question']),
                     'answer': str(row['answer']), 'chain_q': cq, 'chain_a': ca})
    print(f"Data | {len(data)} samples | skipped={skip}")
    return data

def bert_encode_batch(texts, tok, bert, device, max_len=64, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], return_tensors='pt', truncation=True,
                      max_length=max_len, padding=True).to(device)
            out.append(bert(**enc).last_hidden_state[:, 0, :].cpu())
    return torch.cat(out, dim=0)

def precompute_q_embs(data, model_name, cache_path, device):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu')
        print(f"Q-embs loaded ({len(d)} entries)")
        return d
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device); bert.eval()
    d = {}
    for i in tqdm(range(0, len(data), 32), desc='BERT-Q'):
        batch = data[i:i+32]
        enc   = tok([b['question'] for b in batch], return_tensors='pt',
                    truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            cls = bert(**enc).last_hidden_state[:, 0, :]
        for j, e in enumerate(cls): d[batch[j]['idx']] = e.cpu()
    torch.save(d, cache_path); del bert
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return d

def precompute_node_embs(all_nodes, model_name, cache_path, device):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu')
        print(f"Node-embs loaded ({len(d)} nodes)")
        return d
    tok  = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device); bert.eval()
    embs = bert_encode_batch([str(n) for n in all_nodes], tok, bert, device,
                             max_len=32, bs=128)
    d = {node: embs[i] for i, node in enumerate(all_nodes)}
    torch.save(d, cache_path); del bert
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Node-embs cached ({len(d)} nodes)")
    return d

# ─────────────────────── NEW: attention chain aggregation ────────────────────

class ChainAttention(nn.Module):
    """Soft-attention pooling over chain node embeddings (replaces mean pooling)."""
    def __init__(self, dim=768):
        super().__init__()
        self.attn_proj = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_proj.weight)

    def forward(self, node_embs_list, fallback_emb):
        """node_embs_list: list of (768,) tensors. Returns: (768,)."""
        if not node_embs_list:
            return fallback_emb
        stacked = torch.stack(node_embs_list, dim=0)          # (T, 768)
        weights = torch.softmax(
            self.attn_proj(stacked).squeeze(-1), dim=0)        # (T,)
        return (weights.unsqueeze(-1) * stacked).sum(0)        # (768,)

# ─────────────────────── NEW: residual policy network ────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
            nn.init.zeros_(l.bias)

    def forward(self, x):
        r = x
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return F.relu(self.norm(x + r))


class PolicyNetwork(nn.Module):
    """Residual policy network. input_dim=1536 (q_emb‖chain_state)."""
    def __init__(self, output_size, input_dim=768*2, hidden=768, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.res1 = ResidualBlock(hidden, dropout)
        self.res2 = ResidualBlock(hidden, dropout)
        self.neck = nn.Linear(hidden, 256)
        self.out  = nn.Linear(256, output_size)
        self.drop = nn.Dropout(dropout)
        for l in [self.proj, self.neck, self.out]:
            nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
            nn.init.zeros_(l.bias)

    def forward(self, x):
        x = F.relu(self.proj(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.drop(F.relu(self.neck(x)))
        return self.out(x)                      # raw logits

# ─────────────────────── KG helpers ──────────────────────────────────────────

def get_valid_successors(G, q_cot, visited):
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


def compute_reward(node, golden_set, config, G=None):
    if node in golden_set: return 1.0
    if not config['use_fuzzy_reward']: return 0.0
    n_l = node.lower().strip()
    for g in golden_set:
        g_l = g.lower().strip()
        if n_l in g_l or g_l in n_l:
            return config['fuzzy_substring_score']
        n_nums = set(re.findall(r'\d+', n_l))
        g_nums = set(re.findall(r'\d+', g_l))
        if n_nums and n_nums == g_nums:
            return config['fuzzy_number_score']
    # neighbour shaping
    if G is not None:
        shaping = config.get('neighbour_shaping_score', 0.3)
        for nb in list(G.successors(node)) + list(G.predecessors(node)):
            if nb in golden_set:
                return shaping
    return 0.0

# ─────────────────────── GAE ─────────────────────────────────────────────────

def compute_gae(rewards, values, gamma, lam):
    advantages = []; gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae   = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# ─────────────────────── chain state helper ──────────────────────────────────

def get_chain_state(chain_nodes, node_emb_dict, fallback_emb, chain_attn, device):
    """Compute attention-pooled chain state from node embeddings."""
    valid_embs = [node_emb_dict[n].to(device)
                  for n in chain_nodes if n in node_emb_dict]
    return chain_attn(valid_embs, fallback_emb)   # (768,)

# ─────────────────────── training ─────────────────────────────────────────────

def train(config):
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"Device: {device}")

    if config.get('force_fresh'):
        for f in [config['save_path'], config['log_path'],
                  config['emb_cache_path'], config['node_emb_cache_path']]:
            if os.path.exists(f): os.remove(f); print(f"Deleted: {f}")

    G, all_nodes, node2idx, idx2node = load_kg(config['kg_path'])
    all_data      = load_data(config['data_path'], node2idx)
    emb_dict      = precompute_q_embs(all_data, config['bert_model'],
                                       config['emb_cache_path'], device)
    node_emb_dict = precompute_node_embs(all_nodes, config['bert_model'],
                                          config['node_emb_cache_path'], device)

    random.shuffle(all_data)
    n_val      = max(1, int(len(all_data) * config['val_split']))
    val_data   = all_data[:n_val]
    train_data = all_data[n_val:]
    print(f"Train={len(train_data)} | Val={len(val_data)}")

    num_nodes   = len(all_nodes)
    chain_attn  = ChainAttention(dim=768).to(device)
    policy_net  = PolicyNetwork(num_nodes, input_dim=768*2).to(device)
    params      = list(policy_net.parameters()) + list(chain_attn.parameters())
    optimizer   = optim.Adam(params, lr=config['lr'],
                             weight_decay=config['weight_decay'])
    scheduler   = (optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=config['epochs'],
                       eta_min=config['lr'] * 0.01)
                   if config.get('use_lr_scheduler') else None)

    gamma        = config['gamma']
    gae_lam      = config.get('gae_lambda', 0.95)
    ppo_clip     = config.get('ppo_clip_eps', 0.2)
    ppo_epochs   = config.get('ppo_epochs', 3)
    entropy_coef = config['entropy_coef']
    base_decay   = config['baseline_decay']
    temp_init    = config.get('temp_init', 2.0)
    batch_size   = config['batch_size']
    n_batches    = (len(train_data) + batch_size - 1) // batch_size
    best_val     = -float('inf')
    reward_base  = 0.0
    log_rows     = []

    for epoch in range(config['epochs']):
        temperature = max(1.0, temp_init - (temp_init - 1.0) *
                          epoch / max(config['epochs'] - 1, 1))
        random.shuffle(train_data)
        policy_net.train(); chain_attn.train()
        epoch_loss = epoch_rwd = 0.0

        for bj in tqdm(range(n_batches),
                       desc=f"Epoch {epoch+1:02d}/{config['epochs']}", leave=False):
            batch = train_data[bj * batch_size:(bj + 1) * batch_size]

            # ── PHASE 1: collect episodes (no grad — only store raw data) ──
            # We store: state_emb (detached), valid_mask, action_taken (int),
            #           advantage, reward so we can RECOMPUTE log-probs in PPO.
            episodes = []

            with torch.no_grad():
                for sample in batch:
                    q_cot   = copy.deepcopy(sample['chain_q'])
                    a_set   = set(sample['chain_a'])
                    q_emb   = emb_dict[sample['idx']].to(device)   # (768,)
                    visited = set(q_cot)

                    # per-step storage (all detached)
                    step_states  = []   # list of (1536,) tensors — DETACHED
                    step_masks   = []   # list of (num_nodes,) bool tensors
                    step_actions = []   # list of int
                    step_rewards = []
                    step_values  = []

                    for _ in range(config['train_steps']):
                        succs = get_valid_successors(G, q_cot, visited)
                        if not succs: break
                        valid_idx = [node2idx[n] for n in succs if n in node2idx]
                        if not valid_idx: break

                        chain_state = get_chain_state(
                            q_cot, node_emb_dict, q_emb, chain_attn, device)
                        state_emb = torch.cat([q_emb, chain_state], dim=0)  # (1536,)

                        # forward pass to sample action
                        logits  = policy_net(state_emb.unsqueeze(0)).squeeze(0)
                        mask    = torch.full((num_nodes,), float('-inf'), device=device)
                        mask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = 0.0
                        probs   = F.softmax((logits + mask) / temperature, dim=0)
                        action  = Categorical(probs).sample().item()

                        nxt = idx2node[action]
                        rwd = compute_reward(nxt, a_set, config, G)

                        # store DETACHED state + mask for PPO recompute
                        step_states.append(state_emb.detach())
                        # store valid_idx as a boolean mask
                        vmask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                        vmask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = True
                        step_masks.append(vmask)
                        step_actions.append(action)
                        step_rewards.append(rwd)
                        step_values.append(float(np.mean(step_rewards)))

                        q_cot.append(nxt); visited.add(nxt)

                    if not step_actions:
                        continue

                    # compute advantages (GAE or discounted)
                    if config.get('use_gae'):
                        advantages = compute_gae(step_rewards, step_values,
                                                 gamma, gae_lam)
                    else:
                        R, advantages = 0.0, []
                        for r in reversed(step_rewards):
                            R = r + gamma * R; advantages.insert(0, R)
                        advantages = [a - reward_base for a in advantages]

                    reward_base = (base_decay * reward_base
                                   + (1 - base_decay) * sum(step_rewards))
                    episodes.append({
                        'states':    step_states,    # list of detached (1536,)
                        'masks':     step_masks,     # list of bool (num_nodes,)
                        'actions':   step_actions,   # list of int
                        'advantages':advantages,     # list of float
                        'rewards':   step_rewards,
                    })
                    epoch_rwd += sum(step_rewards)

            if not episodes:
                continue

            # ── PHASE 2: PPO update (new graph each inner epoch) ──────────
            for ppo_epoch in range(ppo_epochs if config.get('use_ppo') else 1):
                total_loss = torch.tensor(0.0, device=device)
                n_valid    = 0

                for ep in episodes:
                    if not ep['states']:
                        continue

                    adv_t = torch.tensor(ep['advantages'],
                                         dtype=torch.float32, device=device)
                    # normalise advantages
                    if adv_t.numel() > 1:
                        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                    ep_policy_loss  = torch.tensor(0.0, device=device)
                    ep_entropy_loss = torch.tensor(0.0, device=device)

                    for t, (state_emb, vmask, action, adv) in enumerate(
                            zip(ep['states'], ep['masks'],
                                ep['actions'], adv_t)):

                        # ── FRESH forward pass — builds a NEW graph ──────
                        logits    = policy_net(state_emb.unsqueeze(0)).squeeze(0)
                        neg_inf   = torch.full_like(logits, float('-inf'))
                        logits_m  = torch.where(vmask, logits, neg_inf)
                        probs     = F.softmax(logits_m, dim=0)
                        dist      = Categorical(probs)
                        action_t  = torch.tensor(action, device=device)

                        new_log_p = dist.log_prob(action_t)
                        entropy   = dist.entropy()

                        if config.get('use_ppo'):
                            # PPO clip — old_log_p is detached (from collect phase)
                            # We recompute old_log_p here with no grad as reference
                            with torch.no_grad():
                                old_probs    = F.softmax(logits_m, dim=0)
                                old_log_p    = Categorical(old_probs).log_prob(action_t)
                            ratio  = torch.exp(new_log_p - old_log_p)
                            surr1  = ratio * adv
                            surr2  = torch.clamp(ratio, 1 - ppo_clip,
                                                  1 + ppo_clip) * adv
                            step_loss = -torch.min(surr1, surr2)
                        else:
                            step_loss = -new_log_p * adv

                        ep_policy_loss  = ep_policy_loss  + step_loss
                        ep_entropy_loss = ep_entropy_loss + entropy

                    n_steps = len(ep['states'])
                    total_loss = (total_loss
                                  + ep_policy_loss  / n_steps
                                  + entropy_coef * (-ep_entropy_loss / n_steps))
                    n_valid += 1

                if n_valid == 0:
                    continue

                total_loss = total_loss / n_valid
                optimizer.zero_grad()
                total_loss.backward()       # ✅ fresh graph each time — no double-backward
                torch.nn.utils.clip_grad_norm_(params, config['grad_clip'])
                optimizer.step()
                epoch_loss += total_loss.item()

        if scheduler:
            scheduler.step()

        # ── validation ────────────────────────────────────────────────────
        policy_net.eval(); chain_attn.eval()
        val_rwd = 0.0
        with torch.no_grad():
            for sample in val_data:
                q_cot   = copy.deepcopy(sample['chain_q'])
                a_set   = set(sample['chain_a'])
                q_emb   = emb_dict[sample['idx']].to(device)
                visited = set(q_cot)
                for _ in range(config['test_steps']):
                    succs = get_valid_successors(G, q_cot, visited)
                    if not succs: break
                    valid_idx = [node2idx[n] for n in succs if n in node2idx]
                    if not valid_idx: break
                    chain_state = get_chain_state(
                        q_cot, node_emb_dict, q_emb, chain_attn, device)
                    state_emb = torch.cat([q_emb, chain_state], dim=0)
                    logits = policy_net(state_emb.unsqueeze(0)).squeeze(0)
                    mask   = torch.full((num_nodes,), float('-inf'), device=device)
                    mask[torch.tensor(valid_idx, dtype=torch.long, device=device)] = 0.0
                    probs  = F.softmax(logits + mask, dim=0)
                    action = Categorical(probs).sample().item()
                    nxt    = idx2node[action]
                    val_rwd += compute_reward(nxt, a_set, config, G)
                    q_cot.append(nxt); visited.add(nxt)

        avg_loss = epoch_loss / max(n_batches * ppo_epochs, 1)
        avg_rwd  = epoch_rwd  / max(len(train_data), 1)
        avg_vrd  = val_rwd    / max(len(val_data), 1)
        cur_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss:{avg_loss:.4f} | TrainR:{avg_rwd:.4f} | "
              f"ValR:{avg_vrd:.4f} | LR:{cur_lr:.2e} | T:{temperature:.2f}")

        log_rows.append({'epoch': epoch+1, 'loss': round(avg_loss, 4),
                         'train_reward': round(avg_rwd, 4),
                         'val_reward':   round(avg_vrd, 4),
                         'lr':           round(cur_lr, 8),
                         'temperature':  round(temperature, 3)})
        pd.DataFrame(log_rows).to_csv(config['log_path'], index=False)

        if avg_vrd > best_val:
            best_val = avg_vrd
            torch.save({
                'epoch':                  epoch + 1,
                'policy_state_dict':      policy_net.state_dict(),
                'chain_attn_state_dict':  chain_attn.state_dict(),
                'optimizer_state_dict':   optimizer.state_dict(),
                'config':                 config,
                'best_val_reward':        best_val,
                'all_nodes':              all_nodes,
                'node2idx':               node2idx,
                'idx2node':               idx2node,
            }, config['save_path'])
            print(f"  ✓ Best model saved | val_reward={best_val:.4f}")

    print(f"\nTraining complete | model={config['save_path']}")
    return policy_net, chain_attn


if __name__ == '__main__':
    train(CONFIG)
