"""
Multi-Domain Modern Statutory Legal QA Batch Runner (`run_bns_queries.py`)
========================================================================

Acts as the statutory conversion layer after the legal retrieval/reasoning pipeline
has been run and before feeding context to the final GPT-4o-mini model.

If any legacy IPC (or IEA) sections are present in:
  - The user's question
  - Retrieved historical case precedents
  - Knowledge graph reasoning chain paths (via 2-hop traversal)

They are mapped to their corresponding modern statutory provisions:
  - Indian Penal Code (IPC, 1860) -> Bharatiya Nyaya Sanhita (BNS, 2023)
  - Indian Evidence Act (IEA, 1872) -> Bharatiya Sakshya Adhiniyam (BSA, 2023)

Usage:
    # Process an input CSV with questions and output a CSV with answers:
    python ipc_to_bns/run_bns_queries.py --input questions.csv --output answers.csv

    # Specify mode or domain:
    python ipc_to_bns/run_bns_queries.py --input questions.csv --output answers.csv --mode actionable --domain labour

    # Run with default built-in sample questions if no input CSV specified:
    python ipc_to_bns/run_bns_queries.py
"""

import os
import sys
import re
import argparse
import logging
from typing import Optional, List, Dict

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure repository root and ipc_to_bns directory are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
for p in (_REPO_ROOT, _CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from ipc_to_bns.bns_converter import (
    BNSConverter,
    extract_statutes_from_text,
    load_statutory_mappings,
    load_node_metadata,
    build_domain_prompt,
    build_statutory_candidate_menu,
    extract_2hop_statutory_sections,
)
from inference.config import INFERENCE_CONFIG
from inference.engine import LegalQAEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_bns_queries")


# ── Sample Questions Dataset ──────────────────────────────────────────────────
# Default sample dataset covering disputes where IPC / BNS sections are relevant
SAMPLE_QUESTIONS = [
    {
        "category": "Cheating & Financial Fraud",
        "question": "Can my employer be criminally charged under Section 420 IPC for cheating if they fraudulently induced me to sign a bond with false salary promises?",
    },
    {
        "category": "Criminal Breach of Trust",
        "question": "My company deducted Provident Fund contributions from my wages for 8 months but failed to deposit them with EPFO. Can the directors be booked under Section 406 IPC for criminal breach of trust?",
    },
    {
        "category": "Workplace Harassment & Cruelty",
        "question": "My colleague and supervisor have been physically and mentally harassing me at the company office, threatening false complaints. Does Section 498A IPC apply or what criminal charges can be filed?",
    },
    {
        "category": "Criminal Intimidation",
        "question": "The contractor threatened to kill me and beat up my family when I demanded my pending wages. What action can be taken for criminal intimidation under IPC 506?",
    },
    {
        "category": "Wage Non-Payment & Recovery",
        "question": "My employer has not paid my salary for the last four months and is withholding my experience certificate. What are my legal remedies to recover my dues?",
    },
    {
        "category": "Wrongful Confinement",
        "question": "The manager locked workers inside the factory floor after shift hours to prevent us from protesting against safety violations. Can this be prosecuted under IPC Section 342 for wrongful confinement?",
    },
]


def resolve_domain_data_dir(domain: str, mode: str = "actionable") -> str:
    """Resolves data directory containing checkpoints and graph for given domain."""
    domain_map = {
        "labour": ("HKG+PC+RL", "Employment_Labour", mode),
        "employment": ("HKG+PC+RL", "Employment_Labour", mode),
        "criminal": ("HKG+PC+RL", "criminal", mode),
        "family": ("HKG+PC+RL", "family", mode),
    }

    subpath = domain_map.get(domain.lower(), ("HKG+PC+RL", "Employment_Labour", mode))
    target_dir = os.path.join(_REPO_ROOT, *subpath)

    # Fallback to Employment_Labour if domain directory lacks checkpoints
    rl_pt = os.path.join(target_dir, "rl_policy_model.pt")
    if not os.path.isfile(rl_pt):
        fallback_dir = os.path.join(_REPO_ROOT, "HKG+PC+RL", "Employment_Labour", mode)
        if os.path.isfile(os.path.join(fallback_dir, "rl_policy_model.pt")):
            logger.info("Using fallback data directory with active checkpoints: %s", fallback_dir)
            return fallback_dir

    return target_dir


def create_engine_for_bns(domain: str = "labour", mode: str = "actionable", model: str = "gpt-4o-mini") -> LegalQAEngine:
    """Builds and loads an engine configured with the BNS statutory conversion layer."""
    import copy
    cfg = copy.deepcopy(INFERENCE_CONFIG)

    data_dir = resolve_domain_data_dir(domain, mode)
    cfg["LEGAL_QA_DATA_DIR"] = data_dir
    cfg["rl_model_path"] = os.path.join(data_dir, "rl_policy_model.pt")
    cfg["dssm_model_path"] = os.path.join(data_dir, "dssm_model.pt")
    cfg["retrieval_index"] = os.path.join(data_dir, "dssm_retrieval_index.pt")
    cfg["node_emb_cache_path"] = os.path.join(data_dir, "node_embeddings.pt")
    cfg["kg_path"] = os.path.join(data_dir, "edges.csv")
    cfg["nodes_csv_path"] = os.path.join(data_dir, "nodes.csv")

    cfg["enable_bns_conversion"] = True
    cfg["bns_mapping_csv"] = os.path.join(_CURRENT_DIR, "bns_ipc_mapping_clean.csv")
    cfg["domain_name"] = "Employment & Labour Law" if "labour" in domain.lower() else (
        "Criminal Law" if "crim" in domain.lower() else "Family & Matrimonial Law"
    )
    # This runner is domain-specific: keep the chosen domain instead of auto-detecting per question
    cfg["auto_detect_domain"] = False

    if model:
        cfg["openai_model"] = model

    engine = LegalQAEngine(cfg)
    logger.info("Loading models for %s [%s]...", domain, mode)
    engine.load_models()
    return engine


def find_question_column(columns: List[str]) -> str:
    """Identifies the question column in an input CSV."""
    candidates = ["question", "query", "query_text", "prompt", "legal_question", "text"]
    col_lower = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    # Default to first column if no named candidate found
    return columns[0]


def process_bns_csv(
    input_csv: str,
    output_csv: str,
    domain: str = "labour",
    mode: str = "actionable",
    model: str = "gpt-4o-mini",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reads an input CSV of questions, runs the legal QA pipeline with BNS statutory conversion,
    and writes results with modern BNS citations to the output CSV.
    """
    # 1. Load input questions
    if not os.path.isfile(input_csv):
        logger.info("Input CSV '%s' not found. Creating sample input CSV...", input_csv)
        df_sample = pd.DataFrame(SAMPLE_QUESTIONS)
        os.makedirs(os.path.dirname(os.path.abspath(input_csv)) or ".", exist_ok=True)
        df_sample.to_csv(input_csv, index=False)
        df = df_sample
    else:
        df = pd.read_csv(input_csv)

    if limit is not None and limit > 0:
        df = df.head(limit)

    q_col = find_question_column(list(df.columns))
    logger.info("Identified question column: '%s' (total rows: %d)", q_col, len(df))

    # 2. Initialize engine with BNS layer
    engine = create_engine_for_bns(domain=domain, mode=mode, model=model)

    records = []
    print("\n" + "=" * 70)
    print(f" Running Legal QA Pipeline with IPC->BNS Statutory Layer")
    print(f" Input:  {input_csv} ({len(df)} questions)")
    print(f" Output: {output_csv}")
    print(f" Model:  {engine.config.get('openai_model', 'gpt-4o-mini')}")
    print("=" * 70 + "\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = str(row[q_col]).strip()
        if not question or question.lower() == "nan":
            continue

        try:
            res = engine.predict(question)
            answer = res.get("answer", "")
            chain = res.get("reasoning_chain", [])
            bns_data = res.get("bns_data", {})

            discovered_ipcs = ", ".join(bns_data.get("discovered_ipcs", []))
            mapped_bns = ", ".join(bns_data.get("mapped_bns_sections", []))
            statutory_menu = bns_data.get("statutory_menu", "")
            has_conversion = bns_data.get("has_ipc_conversion", False)

        except Exception as e:
            logger.error("Error processing question '%s': %s", question[:40], e)
            answer = f"ERROR: {e}"
            chain = []
            discovered_ipcs = ""
            mapped_bns = ""
            statutory_menu = ""
            has_conversion = False

        record = dict(row)
        record["generated_modern_answer"] = answer
        record["rl_reasoning_chain"] = " → ".join(chain) if chain else "N/A"
        record["discovered_legacy_ipcs"] = discovered_ipcs or "None"
        record["mapped_modern_bns_sections"] = mapped_bns or "None"
        record["ipc_to_bns_converted"] = has_conversion
        record["candidate_provisions_provided"] = statutory_menu.replace("\n", " | ")
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SUCCESS] Processing complete! Saved {len(out_df)} results to '{output_csv}'.\n")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Run batch legal QA queries through the pipeline with IPC-to-BNS statutory conversion."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join(_CURRENT_DIR, "my_bns_questions.csv"),
        help="Path to input CSV containing questions (default: ipc_to_bns/my_bns_questions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join(_CURRENT_DIR, "bns_test_answers.csv"),
        help="Path to output CSV for answers (default: ipc_to_bns/bns_test_answers.csv)"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        default="labour",
        choices=["labour", "criminal", "family"],
        help="Domain knowledge graph (default: labour)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="actionable",
        choices=["actionable", "informative", "readable"],
        help="Inference tuning mode (default: actionable)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for answer generation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of questions to process"
    )

    args = parser.parse_args()

    process_bns_csv(
        input_csv=args.input,
        output_csv=args.output,
        domain=args.domain,
        mode=args.mode,
        model=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()