"""
Centralized configuration for the Legal_QA inference API.

All paths default to the flagship HKG+PC+RL / Employment_Labour / actionable
experiment directory.  Override via environment variables or by editing
DEFAULT_DATA_DIR below.
"""

import os

# Configure HuggingFace to use a local cache directory inside the repository
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(_REPO_ROOT, ".hf_cache")

import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Default data directory (flagship experiment) ────────────────────────────
DEFAULT_DATA_DIR = os.environ.get(
    "LEGAL_QA_DATA_DIR",
    os.path.join(_REPO_ROOT, "HKG+PC+RL", "Employment_Labour", "actionable"),
)


def _data_path(filename: str) -> str:
    """Resolve a filename relative to the data directory."""
    return os.path.join(DEFAULT_DATA_DIR, filename)


# ── Inference configuration ─────────────────────────────────────────────────
INFERENCE_CONFIG = {
    # ── paths (resolved to the data directory) ──────────────────────────────
    "rl_model_path":        _data_path("rl_policy_model.pt"),
    "dssm_model_path":      _data_path("dssm_model.pt"),
    "retrieval_index":      _data_path("dssm_retrieval_index.pt"),
    "node_emb_cache_path":  _data_path("node_embeddings.pt"),
    "kg_path":              _data_path("edges.csv"),
    "nodes_csv_path":       _data_path("nodes.csv"),

    # ── retrieval ───────────────────────────────────────────────────────────
    "top_k":                3,
    "dssm_candidate_k":     5,
    "use_cross_encoder_rerank": False,
    "use_bm25_fallback":    True,
    "mmr_lambda":           0.6,

    # ── RL inference ────────────────────────────────────────────────────────
    "rl_test_steps":        2,
    "rl_greedy":            True,   # argmax over valid successors (deterministic); False = sample
    "use_chain_query_expand": True,

    # ── seed chain discovery (TF-IDF over KG node labels) ───────────────────
    "seed_top_k_candidates": 30,
    "seed_max_chain_len":    4,

    # ── LLM ─────────────────────────────────────────────────────────────────
    "llm_backend":          os.environ.get("LLM_BACKEND", "openai"),
    "groq_api_key":         os.environ.get("GROQ_API_KEY", ""),
    "groq_model":           os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "openai_api_key":       os.environ.get("OPENAI_API_KEY", ""),
    "openai_model":         os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "ollama_model":         os.environ.get("OLLAMA_MODEL", "llama3"),
    "ollama_url":           os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "hf_model":             os.environ.get("HF_MODEL", "microsoft/phi-2"),
    "temperature":          0.4,
    "top_p":                0.9,
    "max_new_tokens":       200,

    # ── IPC to BNS statutory transition layer ─────────────────────────────
    "enable_bns_conversion": True,
    "bns_mapping_csv":       os.path.join(_REPO_ROOT, "ipc_to_bns", "bns_ipc_mapping_clean.csv"),
    "domain_name":           os.environ.get("LEGAL_DOMAIN", "Employment & Labour Law"),
    # Classify each question as criminal / family / labour for the prompt (domain_name is the fallback)
    "auto_detect_domain":    os.environ.get("AUTO_DETECT_DOMAIN", "true").strip().lower() in ("1", "true", "yes"),
    # Whole-BNS InLegalBERT similarity search for candidate sections. Off: raw [CLS]
    # similarities are not discriminative and added unrelated sections to prompts.
    "semantic_statute_search": os.environ.get("SEMANTIC_STATUTE_SEARCH", "false").strip().lower() in ("1", "true", "yes"),
    # CrPC -> BNSS concordance (verified against the CAPT Bhopal correspondence table)
    "crpc_mapping_csv":      os.path.join(_REPO_ROOT, "ipc_to_bns", "crpc_bnss_mapping.csv"),

    # System prompt for the LLM; overrides SYSTEM_PROMPT in each domain's infer.py
    "system_prompt": (
        "You are an experienced Indian legal advocate advising a client. "
        "Provide specific, accurate, actionable legal advice under modern Indian law. "
        "CRITICAL MANDATE: In India, the Indian Penal Code (IPC), 1860 has been officially superseded and repealed "
        "by the Bharatiya Nyaya Sanhita (BNS), 2023, and the Indian Evidence Act (IEA), 1872 by the Bharatiya Sakshya Adhiniyam (BSA), 2023. "
        "Under no circumstances should you cite the IPC or IEA for criminal offenses. "
        "Always cite the applicable modern statutes: the Bharatiya Nyaya Sanhita (BNS), 2023 for penal matters, "
        "the Bharatiya Sakshya Adhiniyam (BSA), 2023 for evidence, and applicable special acts (e.g. IT Act 2000). "
        "Keep answers concise (3-5 sentences). "
        "Do NOT add disclaimers about consulting a lawyer — the user already knows."
    ),

    # ── misc ────────────────────────────────────────────────────────────────
    "bert_model":           "law-ai/InLegalBERT",
    "seed":                 42,
    "device":               "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Domains ─────────────────────────────────────────────────────────────────
# API domain key -> (folder under HKG+PC+RL/, domain label used in the prompt).
# Labels match ipc_to_bns.bns_converter.{LABOUR,CRIMINAL,FAMILY}_DOMAIN.
DOMAINS = {
    "labour":   ("Employment_Labour", "Employment & Labour Law"),
    "criminal": ("criminal",          "Criminal Law"),
    "family":   ("family",            "Family & Matrimonial Law"),
}
MODES = ("actionable", "informative", "readable")
FALLBACK_DOMAIN = "labour"


def data_dir_for(domain: str, mode: str) -> str:
    """Absolute path of the HKG+PC+RL experiment folder for a domain/mode."""
    return os.path.join(_REPO_ROOT, "HKG+PC+RL", DOMAINS[domain][0], mode)


def make_config(domain: str, mode: str) -> dict:
    """INFERENCE_CONFIG with every data path pointed at the given domain/mode folder."""
    import copy
    d = data_dir_for(domain, mode)
    cfg = copy.deepcopy(INFERENCE_CONFIG)
    cfg.update({
        "LEGAL_QA_DATA_DIR":   d,
        "rl_model_path":       os.path.join(d, "rl_policy_model.pt"),
        "dssm_model_path":     os.path.join(d, "dssm_model.pt"),
        "retrieval_index":     os.path.join(d, "dssm_retrieval_index.pt"),
        "node_emb_cache_path": os.path.join(d, "node_embeddings.pt"),
        "kg_path":             os.path.join(d, "edges.csv"),
        "nodes_csv_path":      os.path.join(d, "nodes.csv"),
        # The engine serves exactly this domain, so the prompt label is fixed to it.
        "domain_name":         DOMAINS[domain][1],
        "auto_detect_domain":  False,
    })
    return cfg


def get_required_files(config: dict | None = None) -> list[dict]:
    """Return the files an engine needs, resolved against `config` (default: INFERENCE_CONFIG)."""
    c = config or INFERENCE_CONFIG
    return [
        {
            "path": c["rl_model_path"],
            "name": "rl_policy_model.pt",
            "description": "Trained PPO RL policy model (generated by train_rl.py)",
        },
        {
            "path": c["dssm_model_path"],
            "name": "dssm_model.pt",
            "description": "Trained DSSM retrieval model (generated by train_dssm.py)",
        },
        {
            "path": c["retrieval_index"],
            "name": "dssm_retrieval_index.pt",
            "description": "Pre-computed retrieval embeddings and DB data (generated by train_dssm.py)",
        },
        {
            "path": c["node_emb_cache_path"],
            "name": "node_embeddings.pt",
            "description": "Pre-computed BERT embeddings for KG nodes (generated during training)",
        },
        {
            "path": c["kg_path"],
            "name": "edges.csv",
            "description": "Hierarchical Knowledge Graph edge list",
        },
        {
            "path": c["nodes_csv_path"],
            "name": "nodes.csv",
            "description": "KG node metadata (used for seed chain discovery via TF-IDF)",
        },
    ]
