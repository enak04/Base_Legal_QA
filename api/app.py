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
import torch
from typing import Literal, Optional

# Configure PyTorch CPU threading for optimal matrix operations without thread contention
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

from inference.config import DOMAINS, MODES, FALLBACK_DOMAIN, make_config, get_required_files
from inference.engine import LegalQAEngine
from ipc_to_bns.bns_converter import detect_legal_domain

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("legal_qa_api")

# ── Dynamic Config Construction ──────────────────────────────────────────────
def make_config_for_mode(mode: str, domain: str = FALLBACK_DOMAIN) -> dict:
    """Config for one HKG+PC+RL domain/mode folder (domain defaults to labour)."""
    return make_config(domain, mode)


def is_available(domain: str, mode: str) -> bool:
    """True if every checkpoint/data file for this domain/mode exists on disk."""
    return all(os.path.isfile(f["path"]) for f in get_required_files(make_config(domain, mode)))


# ── Engines Dictionary ──────────────────────────────────────────────────────
# Keyed by (domain, mode), e.g. ("criminal", "actionable").
engines: dict[tuple[str, str], LegalQAEngine] = {}

# ── Shared InLegalBERT instances ───────────────────────────────────────────
shared_tok = None
shared_bert = None

def get_engine(domain: str, mode: str) -> LegalQAEngine:
    global shared_tok, shared_bert

    # If the engine is already loaded, return it immediately from cache
    key = (domain, mode)
    if key in engines:
        return engines[key]

    logger.info("Loading and caching engine for %s/%s...", domain, mode)
    cfg = make_config(domain, mode)
    eng = LegalQAEngine(cfg)

    # Check that required files are present
    missing = eng.check_required_files()
    if missing:
        msg_lines = [f"Missing required checkpoints for {domain}/{mode}:"]
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
    engines[key] = eng
    return eng


def get_engine_for_mode(mode: str) -> LegalQAEngine:
    """Backward-compatible alias: the labour engine for a mode."""
    return get_engine(FALLBACK_DOMAIN, mode)


def resolve_engine(requested_domain: str, question: str, mode: str):
    """
    Pick the engine for a request.

    Returns (engine, domain_detected, domain_used). `domain_detected` is the
    requested domain, or the keyword classifier's guess when "auto". If that
    domain has no trained checkpoints yet, the labour engine serves it
    (`domain_used` = "labour") and the prompt keeps the detected domain's label.
    """
    if requested_domain == "auto":
        label = detect_legal_domain(question, default=DOMAINS[FALLBACK_DOMAIN][1])
        domain = next(k for k, (_, lbl) in DOMAINS.items() if lbl == label)
    else:
        domain = requested_domain

    if domain != FALLBACK_DOMAIN and not is_available(domain, mode):
        logger.info("No trained checkpoints for %s/%s — falling back to %s.", domain, mode, FALLBACK_DOMAIN)
        return get_engine(FALLBACK_DOMAIN, mode), domain, FALLBACK_DOMAIN
    return get_engine(domain, mode), domain, domain

# ── Lifespan (startup pre-warming) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads every domain/mode that has trained checkpoints, so queries never wait on loading."""
    logger.info("=" * 65)
    logger.info("  Legal QA API — Starting & pre-loading all available domain/mode engines...")
    logger.info("=" * 65)

    for domain in DOMAINS:
        for mode in MODES:
            if not is_available(domain, mode):
                logger.warning("  ⚠️ %s/%s: no trained checkpoints — requests fall back to %s.",
                               domain, mode, FALLBACK_DOMAIN)
                continue
            try:
                logger.info("  → Loading %s/%s...", domain, mode)
                get_engine(domain, mode)
            except Exception as e:
                logger.warning("  ⚠️ Could not load %s/%s: %s", domain, mode, e)

    logger.info("=" * 65)
    logger.info("  ✅ Loaded engines: %s", ", ".join(f"{d}/{m}" for d, m in engines) or "none")
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
    domain: Literal["auto", "labour", "criminal", "family"] = Field(
        "auto",
        description=(
            "Legal domain whose knowledge graph and models to use. 'auto' classifies the "
            "question by keywords; pass the domain explicitly when you know it."
        ),
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
    domain: str = Field(..., description="Domain of the question (requested, or detected when 'auto').")
    domain_used: str = Field(..., description="Domain whose knowledge graph and models produced the chain and cases.")
    domain_fallback: bool = Field(False, description="True when the question's domain has no trained models yet and the labour models were used.")


class HealthResponse(BaseModel):
    status: str


class DomainStatus(BaseModel):
    domain: str
    mode: str
    available: bool = Field(..., description="Trained checkpoints are present on disk.")
    loaded: bool = Field(..., description="Engine is loaded in memory.")


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


@app.get(
    "/domains",
    response_model=list[DomainStatus],
    summary="Which domain/mode engines have trained models and are loaded",
)
async def domains():
    return [
        {"domain": d, "mode": m, "available": is_available(d, m), "loaded": (d, m) in engines}
        for d in DOMAINS for m in MODES
    ]


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

    # ── Route to the engine for this domain/mode ────────────────────────
    mode = request.mode
    logger.info("📩 Incoming Query [Domain: %s | Mode: %s]: %s", request.domain, mode, question)
    try:
        engine, domain, domain_used = resolve_engine(request.domain, question, mode)
    except Exception as e:
        logger.error("Failed to load engine for %s/%s: %s", request.domain, mode, e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Domain '{request.domain}' / mode '{mode}' could not be loaded: {e}",
        )

    # ── Run inference (offloaded to threadpool to keep event loop responsive) ───
    try:
        # Prompt label is always the question's domain, even when another domain's engine serves it.
        result = await asyncio.to_thread(engine.predict, question, DOMAINS[domain][1])
        result.update(domain=domain, domain_used=domain_used, domain_fallback=(domain != domain_used))
        chain_summary = " -> ".join(result.get("reasoning_chain", []))
        logger.info("✅ Query Completed [%s via %s/%s] | Chain: %s", domain, domain_used, mode, chain_summary)
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
