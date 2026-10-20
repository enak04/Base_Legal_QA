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

# Configure HuggingFace to use a local cache directory inside the repository
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(repo_root, ".hf_cache")

import asyncio
import copy
import torch
from typing import Literal, Optional

# Configure PyTorch CPU threading for optimal matrix operations without thread contention
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

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

    # If the engine is already loaded, return it immediately from cache
    if mode in engines:
        return engines[mode]

    logger.info(f"Loading and caching engine for mode: {mode}...")
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
        
        # Apply 8-bit dynamic quantization to save ~330MB of RAM on CPU
        if device.type == "cpu":
            logger.info("Applying 8-bit dynamic quantization to InLegalBERT for CPU...")
            shared_bert = torch.quantization.quantize_dynamic(
                shared_bert, {torch.nn.Linear}, dtype=torch.qint8
            )

    # Load engine models using the shared BERT instances
    eng.load_models(shared_tok=shared_tok, shared_bert=shared_bert)
    engines[mode] = eng
    return eng

# ── Lifespan (startup pre-warming) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warms all 3 modes on startup so every query is fast with 0s swap delay."""
    logger.info("=" * 65)
    logger.info("  Legal QA API — Starting & pre-warming all 3 modes into RAM...")
    logger.info("=" * 65)

    for mode in ["actionable", "informative", "readable"]:
        try:
            logger.info("  → Pre-warming '%s' mode...", mode)
            get_engine_for_mode(mode)
        except Exception as e:
            logger.warning("  ⚠️ Could not pre-warm mode '%s': %s", mode, e)

    logger.info("=" * 65)
    logger.info("  ✅ All 3 modes pre-warmed & ready for instantaneous inference!")
    logger.info("=" * 65)

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


class BNSData(BaseModel):
    discovered_ipcs: list[str] = Field(default_factory=list, description="Legacy IPC sections found in query or context")
    mapped_bns_sections: list[str] = Field(default_factory=list, description="Corresponding modern BNS provisions mapped")
    statutory_menu: str = Field("", description="Modern candidate statutory menu provided to LLM")
    has_ipc_conversion: bool = Field(False, description="Whether legacy IPC sections were detected and converted")


class PredictResponse(BaseModel):
    question: str
    answer: str
    reasoning_chain: list[str]
    retrieved_cases: list[RetrievedCase]
    bns_data: Optional[BNSData] = Field(None, description="IPC to BNS conversion details if applicable")


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
    logger.info("📩 Incoming Query [Mode: %s]: %s", mode, question)
    try:
        engine = get_engine_for_mode(mode)
    except Exception as e:
        logger.error("Failed to load engine for mode %s: %s", mode, e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Mode '{mode}' could not be loaded: {e}",
        )

    # ── Run inference (offloaded to threadpool to keep event loop responsive) ───
    try:
        result = await asyncio.to_thread(engine.predict, question)
        chain_summary = " -> ".join(result.get("reasoning_chain", []))
        logger.info("✅ Query Completed [Mode: %s] | Chain: %s", mode, chain_summary)
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
