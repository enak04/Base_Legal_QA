"""
Legal QA REST API
=================

FastAPI application that exposes the existing HKG+PC+RL inference pipeline
as a standalone service.

Endpoints:
    GET  /health   → {"status": "ok"}
    POST /predict  → {"question", "answer", "reasoning_chain", "retrieved_cases"}

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import traceback
from contextlib import asynccontextmanager
import gc

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import os
import copy
import torch
from typing import Literal

from inference.config import INFERENCE_CONFIG, get_required_files
from inference.engine import LegalQAEngine

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("legal_qa_api")

# ── Dynamic Config Construction ──────────────────────────────────────────────
def make_config_for_mode(mode: str) -> dict:
    cfg = copy.deepcopy(INFERENCE_CONFIG)
    
    # Resolve the data directory for the specific mode
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mode_dir = os.path.join(repo_root, "HKG+PC+RL", "Employment_Labour", mode)
    
    cfg["LEGAL_QA_DATA_DIR"] = mode_dir
    
    def _data_path(filename: str) -> str:
        return os.path.join(mode_dir, filename)
        
    cfg["rl_model_path"] = _data_path("rl_policy_model.pt")
    cfg["dssm_model_path"] = _data_path("dssm_model.pt")
    cfg["retrieval_index"] = _data_path("dssm_retrieval_index.pt")
    cfg["node_emb_cache_path"] = _data_path("node_embeddings.pt")
    cfg["kg_path"] = _data_path("edges.csv")
    cfg["nodes_csv_path"] = _data_path("nodes.csv")
    
    return cfg

# ── Engines Dictionary ──────────────────────────────────────────────────────
engines: dict[str, LegalQAEngine] = {}

# ── Shared InLegalBERT instances ───────────────────────────────────────────
shared_tok = None
shared_bert = None

def get_engine_for_mode(mode: str) -> LegalQAEngine:
    global shared_tok, shared_bert

    # If the engine is already loaded, return it
    if mode in engines:
        return engines[mode]

    # Otherwise, unload any other loaded engines to save RAM
    if engines:
        logger.info("Unloading existing engine modes to free memory...")
        engines.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Loading engine for mode: {mode}...")
    cfg = make_config_for_mode(mode)
    eng = LegalQAEngine(cfg)

    # Check that required files are present
    missing = eng.check_required_files()
    if missing:
        msg_lines = [f"Missing required checkpoints for mode {mode}:"]
        for f in missing:
            msg_lines.append(f"  ✗ {f['name']} (Path: {f['path']})")
        raise FileNotFoundError("\n".join(msg_lines))

    # Initialize shared model/tokenizer if not already loaded
    if shared_tok is None or shared_bert is None:
        logger.info("Initializing globally shared InLegalBERT...")
        from inference.engine import _load_infer_module
        infer_mod = _load_infer_module(cfg)
        device = torch.device(cfg["device"])
        shared_tok, shared_bert = infer_mod.load_bert(cfg["bert_model"], device)

    # Load engine models using the shared BERT instances
    eng.load_models(shared_tok=shared_tok, shared_bert=shared_bert)
    engines[mode] = eng
    return eng

# ── Lifespan (model loading at startup) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load only the default 'actionable' model at startup to save memory on free tiers."""
    logger.info("=" * 60)
    logger.info("  Legal QA API — Starting up (loading default mode: actionable)")
    logger.info("=" * 60)

    try:
        get_engine_for_mode("actionable")
        logger.info("=" * 60)
        logger.info("  Legal QA API — Ready to serve requests (default mode loaded)")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("  CRITICAL: Default engine mode 'actionable' could not be loaded.")
        logger.error("  Reason: %s", e, exc_info=True)
        logger.error("=" * 60)

    yield  # Server runs here


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal QA API",
    description=(
        "REST API for the HKG+PC+RL Legal Question Answering framework. "
        "Combines Hierarchical Knowledge Graphs, PPO Reinforcement Learning, "
        "DSSM past-case retrieval, cross-encoder reranking, and GPT-4o "
        "generation to produce structured legal answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Request / Response schemas ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The legal question to answer.",
        json_schema_extra={"example": "My employer has not paid my salary for three months."},
    )
    mode: Literal["actionable", "informative", "readable"] = Field(
        "actionable",
        description="The target answer tuning mode.",
    )


class RetrievedCase(BaseModel):
    question: str
    answer: str


class PredictResponse(BaseModel):
    question: str
    answer: str
    reasoning_chain: list[str]
    retrieved_cases: list[RetrievedCase]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str


# ── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that never exposes stack traces or API keys."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred while processing the request."},
    )




# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health():
    """Returns server health status."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Models not loaded"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
    summary="Run legal QA inference",
)
async def predict(request: PredictRequest):
    """
    Run the complete HKG+PC+RL inference pipeline for a single question.

    The pipeline performs:
    1. Seed chain discovery (TF-IDF over KG node labels)
    2. PPO RL reasoning-path expansion on the Hierarchical Knowledge Graph
    3. Two-stage DSSM + cross-encoder past-case retrieval
    4. Structured CoT prompt construction
    5. GPT-4o answer generation
    6. Post-processing
    """
    # ── Validate ────────────────────────────────────────────────────────
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Get or load the requested engine mode ───────────────────────────
    mode = request.mode
    try:
        engine = get_engine_for_mode(mode)
    except Exception as e:
        logger.error("Failed to load engine for mode %s: %s", mode, e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Mode '{mode}' could not be loaded: {e}",
        )

    # ── Run inference ───────────────────────────────────────────────────
    try:
        result = engine.predict(question)
    except Exception as e:
        logger.error("Inference failed for question: %s", question, exc_info=True)

        # Provide a user-friendly message without leaking internals
        error_type = type(e).__name__
        if "openai" in error_type.lower() or "api" in str(e).lower():
            detail = "The language model service encountered an error. Please try again later."
        else:
            detail = "An error occurred during inference. Please try again."

        raise HTTPException(status_code=500, detail=detail)

    return result
