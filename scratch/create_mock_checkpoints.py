import os
import torch
import pandas as pd
import json
import importlib.util
import sys

def generate_checkpoints_for_mode(repo_root, mode):
    data_dir = os.path.join(repo_root, "HKG+PC+RL", "Employment_Labour", mode)
    print(f"\n--- Generating mock checkpoints for mode: {mode} ---")
    print(f"Data directory: {data_dir}")

    kg_path = os.path.join(data_dir, "edges.csv")
    nodes_csv_path = os.path.join(data_dir, "nodes.csv")
    db_data_path = os.path.join(data_dir, "retrieval_database.csv")

    rl_model_path = os.path.join(data_dir, "rl_policy_model.pt")
    dssm_model_path = os.path.join(data_dir, "dssm_model.pt")
    retrieval_index_path = os.path.join(data_dir, "dssm_retrieval_index.pt")
    node_emb_cache_path = os.path.join(data_dir, "node_embeddings.pt")

    # Load infer.py dynamically
    infer_path = os.path.join(data_dir, "infer.py")
    spec = importlib.util.spec_from_file_location(f"_legalqa_infer_{mode}", infer_path)
    mod = importlib.util.module_from_spec(spec)
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    spec.loader.exec_module(mod)

    # 2. Get all nodes from KG
    G, node_set = mod.load_kg(kg_path)
    all_nodes = list(G.nodes)
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    idx2node = {i: n for i, n in enumerate(all_nodes)}

    # 3. Create mock RL Policy Network
    policy = mod.PolicyNetwork(output_size=len(all_nodes), input_dim=768*2)
    chain_attn = mod.ChainAttention(dim=768)

    rl_ckpt = {
        "all_nodes": all_nodes,
        "node2idx": node2idx,
        "idx2node": idx2node,
        "policy_state_dict": policy.state_dict(),
        "chain_attn_state_dict": chain_attn.state_dict(),
        "epoch": 1,
        "best_val_reward": 0.5
    }
    torch.save(rl_ckpt, rl_model_path)
    print("Created mock rl_policy_model.pt")

    # 4. Create mock DSSM model
    dssm = mod.CrossAttentionDSSM(d=768)
    dssm_ckpt = {
        "model_state": dssm.state_dict(),
        "epoch": 1,
        "best_val_acc": 0.5
    }
    torch.save(dssm_ckpt, dssm_model_path)
    print("Created mock dssm_model.pt")

    # 5. Create mock node embeddings (768-dim random tensor for each node)
    node_emb_dict = {}
    for node in all_nodes:
        node_emb_dict[node] = torch.randn(768)
    torch.save(node_emb_dict, node_emb_cache_path)
    print("Created mock node_embeddings.pt")

    # 6. Create mock retrieval index
    db_df = pd.read_csv(db_data_path)
    db_records = []
    for _, row in db_df.iterrows():
        db_records.append({
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", ""))
        })

    num_records = len(db_records)
    hq_tensor = torch.randn(num_records, 768)
    hc_tensor = torch.randn(num_records, 768)

    retrieval_ckpt = {
        "hq": hq_tensor,
        "hc": hc_tensor,
        "db_data": db_records
    }
    torch.save(retrieval_ckpt, retrieval_index_path)
    print("Created mock dssm_retrieval_index.pt")
    print(f"Finished generating mock checkpoints for {mode}!")

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mode in ["actionable", "informative", "readable"]:
        generate_checkpoints_for_mode(repo_root, mode)
    print("\nAll mock checkpoints generated successfully!")

if __name__ == "__main__":
    main()
