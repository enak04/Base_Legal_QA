"""
Legal QA Inference Engine
=========================

Wraps the existing HKG+PC+RL inference pipeline into a reusable class that:
  - Loads all models / checkpoints / data ONCE
  - Provides a `predict(question)` method for single-question inference
  - Discovers seed KG nodes for unseen questions via TF-IDF matching

All model architectures and inference functions are imported directly from
the existing infer.py — no algorithm rewrites.
"""

import os
import sys
import copy
import json
import random
import importlib.util
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic import of the existing infer.py
# ─────────────────────────────────────────────────────────────────────────────
# The research code lives in a directory whose name contains '+' characters
# (e.g. HKG+PC+RL/), making normal Python imports impossible.  We use
# importlib to load the module from its absolute file path instead.

def _load_infer_module(config: dict):
    """
    Dynamically load the existing infer.py as a Python module.
    Returns the module object so we can call its functions.
    """
    infer_path = os.path.join(
        os.path.dirname(config["kg_path"]),  # same dir as edges.csv
        "infer.py",
    )
    if not os.path.isfile(infer_path):
        raise FileNotFoundError(
            f"Cannot find infer.py at {infer_path}. "
            "Ensure the data directory contains the original research code."
        )

    spec = importlib.util.spec_from_file_location("_legalqa_infer", infer_path)
    mod = importlib.util.module_from_spec(spec)

    # Temporarily add the data directory to sys.path so that any relative
    # imports or file references inside infer.py resolve correctly.
    data_dir = os.path.dirname(infer_path)
    original_cwd = os.getcwd()
    try:
        os.chdir(data_dir)
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)

    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Seed chain discovery for unseen questions
# ─────────────────────────────────────────────────────────────────────────────

class SeedChainDiscovery:
    """
    Discovers initial seed KG nodes for a new (unseen) legal question by
    matching the question text against KG node labels using TF-IDF cosine
    similarity.  This replicates the first stage of build_ground_truth.py
    (TF-IDF candidate retrieval) without requiring an LLM call.
    """

    def __init__(self, nodes_csv_path: str, kg_node_set: set,
                 top_k: int = 30, max_chain: int = 4):
        self.top_k = top_k
        self.max_chain = max_chain
        self.kg_node_set = kg_node_set

        # Load node labels from nodes.csv
        df = pd.read_csv(nodes_csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Build a mapping of node_id -> descriptive text for TF-IDF
        self.node_ids = []
        self.node_texts = []
        for _, row in df.iterrows():
            node_id = str(row["node_id"]).strip()
            # Only include nodes that exist in the KG
            if node_id not in kg_node_set:
                continue
            # Use the text field for TF-IDF matching; fall back to node_id
            text = str(row.get("text", node_id)).strip()
            if text in ("nan", ""):
                text = node_id
            self.node_ids.append(node_id)
            self.node_texts.append(f"{node_id} {text}")

        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
        )
        self.node_matrix = self.vectorizer.fit_transform(self.node_texts)
        logger.info(
            "SeedChainDiscovery | indexed %d KG nodes | vocab=%d",
            len(self.node_ids), len(self.vectorizer.vocabulary_),
        )

    def discover(self, question: str) -> list[str]:
        """
        Given a question string, return 1-4 seed KG node IDs that are most
        relevant, filtered to nodes that actually exist in the graph.
        """
        query_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(query_vec, self.node_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:self.top_k]

        # Take the top nodes that exist in the KG
        seed_nodes = []
        for idx in top_indices:
            node_id = self.node_ids[idx]
            if node_id in self.kg_node_set and sims[idx] > 0:
                seed_nodes.append(node_id)
            if len(seed_nodes) >= self.max_chain:
                break

        if not seed_nodes:
            # Absolute fallback: pick any nodes from the KG
            seed_nodes = list(self.kg_node_set)[:2]
            logger.warning(
                "SeedChainDiscovery: no TF-IDF matches for question, "
                "using fallback nodes."
            )

        return seed_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class LegalQAEngine:
    """
    Wraps the complete HKG+PC+RL inference pipeline.

    Usage:
        engine = LegalQAEngine(config)
        engine.load_models()          # call once at startup
        result = engine.predict("My employer has not paid my salary.")
    """

    def __init__(self, config: dict):
        self.config = config
        self._loaded = False

        # These will be populated by load_models()
        self._infer = None           # the dynamically-loaded infer.py module
        self._device = None
        self._G = None               # NetworkX KG
        self._node_set = None
        self._policy = None          # RL policy network
        self._chain_attn = None      # chain attention module
        self._all_nodes = None
        self._node2idx = None
        self._idx2node = None
        self._dssm = None            # DSSM retrieval model
        self._db_hq = None           # retrieval index: question embeddings
        self._db_hc = None           # retrieval index: chain embeddings
        self._db_data = None         # retrieval index: db records
        self._node_emb_dict = None   # pre-computed node embeddings
        self._tok = None             # InLegalBERT tokenizer
        self._bert = None            # InLegalBERT model
        self._seed_chain = None      # SeedChainDiscovery instance

    def check_required_files(self) -> list[dict]:
        """
        Check which required files are missing.
        Returns a list of dicts describing missing files.
        """
        from inference.config import get_required_files
        missing = []
        for f in get_required_files():
            if not os.path.isfile(f["path"]):
                missing.append(f)
        return missing

    def load_models(self, shared_tok=None, shared_bert=None):
        """
        Load all models, checkpoints, KG, embeddings, and indexes.
        This must be called once before predict().
        Raises FileNotFoundError if required checkpoints are missing.
        """
        if self._loaded:
            logger.info("Models already loaded — skipping.")
            return

        config = self.config

        # ── Check required files ────────────────────────────────────────────
        missing = self.check_required_files()
        if missing:
            msg_lines = ["The following required files are missing:"]
            for f in missing:
                msg_lines.append(f"  - {f['name']}: {f['path']}")
                msg_lines.append(f"    ({f['description']})")
            raise FileNotFoundError("\n".join(msg_lines))

        # ── Load the infer.py module ────────────────────────────────────────
        logger.info("Loading infer.py module...")
        self._infer = _load_infer_module(config)

        # ── Set seed and device ─────────────────────────────────────────────
        self._infer.set_seed(config["seed"])
        self._device = torch.device(config["device"])
        logger.info("Device: %s", self._device)

        # ── Load Knowledge Graph ────────────────────────────────────────────
        logger.info("Loading Knowledge Graph from %s...", config["kg_path"])
        self._G, self._node_set = self._infer.load_kg(config["kg_path"])

        # ── Load RL model ───────────────────────────────────────────────────
        logger.info("Loading RL policy model from %s...", config["rl_model_path"])
        (self._policy, self._chain_attn, self._all_nodes,
         self._node2idx, self._idx2node) = self._infer.load_rl_model(
            config["rl_model_path"], self._device
        )

        # ── Load DSSM model ────────────────────────────────────────────────
        logger.info("Loading DSSM model from %s...", config["dssm_model_path"])
        self._dssm = self._infer.load_dssm_model(
            config["dssm_model_path"], self._device
        )

        # ── Load retrieval index ────────────────────────────────────────────
        logger.info("Loading retrieval index from %s...", config["retrieval_index"])
        index = torch.load(config["retrieval_index"], map_location="cpu")
        self._db_hq = index["hq"]
        self._db_hc = index.get("hc", index["hq"])
        self._db_data = index["db_data"]
        logger.info("Retrieval DB: %d records", len(self._db_data))

        # ── Load node embeddings ────────────────────────────────────────────
        logger.info("Loading node embeddings from %s...", config["node_emb_cache_path"])
        self._node_emb_dict = torch.load(
            config["node_emb_cache_path"], map_location="cpu"
        )
        logger.info("Node embeddings: %d nodes cached", len(self._node_emb_dict))

        # ── Load InLegalBERT ────────────────────────────────────────────────
        if shared_tok is not None and shared_bert is not None:
            logger.info("Reusing globally shared InLegalBERT.")
            self._tok = shared_tok
            self._bert = shared_bert
        else:
            logger.info("Loading InLegalBERT (%s)...", config["bert_model"])
            self._tok, self._bert = self._infer.load_bert(
                config["bert_model"], self._device
            )
            # Apply 8-bit dynamic quantization to save ~330MB of RAM on CPU
            if self._device.type == "cpu":
                logger.info("Applying 8-bit dynamic quantization to InLegalBERT for CPU...")
                self._bert = torch.quantization.quantize_dynamic(
                    self._bert, {torch.nn.Linear}, dtype=torch.qint8
                )

        # ── Initialize seed chain discovery ─────────────────────────────────
        logger.info("Building TF-IDF index for seed chain discovery...")
        self._seed_chain = SeedChainDiscovery(
            nodes_csv_path=config["nodes_csv_path"],
            kg_node_set=self._node_set,
            top_k=config.get("seed_top_k_candidates", 30),
            max_chain=config.get("seed_max_chain_len", 4),
        )

        self._loaded = True
        logger.info("All models loaded successfully.")


    def predict(self, question: str) -> dict:
        """
        Run the full inference pipeline for a single question.

        Returns:
            {
                "question": str,
                "answer": str,
                "reasoning_chain": list[str],
                "retrieved_cases": list[{"question": str, "answer": str}]
            }
        """
        if not self._loaded:
            raise RuntimeError(
                "Models not loaded. Call engine.load_models() first."
            )

        infer = self._infer
        config = self.config
        device = self._device

        # Step 1: Discover seed chain_q for this unseen question
        chain_q = self._seed_chain.discover(question)

        # Step 2: Encode question (with optional chain expansion)
        q_emb = infer.encode_text_expanded(
            question, chain_q, self._tok, self._bert, device,
            use_expand=config.get("use_chain_query_expand", True),
        )

        # Step 3: RL chain extension
        chain_extended = infer.predict_chain_rl(
            chain_q, q_emb, self._policy, self._chain_attn,
            self._G, self._all_nodes, self._node2idx, self._idx2node,
            self._node_emb_dict, device,
            steps=config["rl_test_steps"],
        )

        # Step 4: Encode extended chain
        chain_emb = infer.encode_chain_text(
            chain_extended, self._tok, self._bert, device,
        )

        # Step 5: Two-stage retrieval (DSSM → cross-encoder rerank → BM25)
        top_k = infer.retrieve_top_k(
            q_emb, chain_emb, self._dssm,
            self._db_hq, self._db_hc, self._db_data,
            device, config, question, chain_extended,
            self._tok, self._bert,
        )

        # Step 6: Build structured CoT prompt
        prompt = infer.build_prompt(question, chain_extended, top_k)

        # Step 7: Call LLM (GPT-4o by default)
        raw_answer = infer.generate_answer(prompt, config)

        # Step 8: Post-process answer
        answer = infer.postprocess_answer(raw_answer)

        # Step 9: Build response
        retrieved_cases = []
        for case in top_k:
            retrieved_cases.append({
                "question": case.get("question", ""),
                "answer": case.get("answer", ""),
            })

        return {
            "question": question,
            "answer": answer,
            "reasoning_chain": chain_extended,
            "retrieved_cases": retrieved_cases,
        }
