"""
Low-memory InLegalBERT loading for small (e.g. 512 MB) deployments.

Why this exists
---------------
bert-base (InLegalBERT) is ~440 MB in fp32. The original startup path loaded the
fp32 model and then called ``torch.quantization.quantize_dynamic`` without
``inplace=True``, which deep-copies the model, so the process briefly held the
fp32 weights *and* the int8 copy. That spike is what killed the Render instance
("Out of memory (used over 512Mi)" right after "Applying 8-bit dynamic
quantization to InLegalBERT").

Fix
---
1. Build time (Dockerfile):  ``python -m inference.bert_loader``
   loads the fp32 model once, quantizes it in place and saves the int8 model
   (~180 MB) to ``QUANTIZED_BERT_PATH``.
2. Runtime: ``load_bert_low_mem`` loads that file directly, so the fp32 model
   never exists in the server process. If the file is missing or was built with
   different torch / transformers versions, it falls back to loading fp32 and
   quantizing *in place* (no second copy).

The base research code (HKG+PC+RL/**/infer.py) is not modified; the model
returned here is the same ``BertModel`` that ``infer.load_bert`` +
``quantize_dynamic`` produced before, so the encoders in infer.py work unchanged.
"""

import gc
import logging
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_REPO_ROOT, ".hf_cache"))

import torch  # noqa: E402

logger = logging.getLogger(__name__)

QUANTIZED_BERT_PATH = os.environ.get(
    "QUANTIZED_BERT_PATH",
    os.path.join(_REPO_ROOT, "models", "inlegalbert_qint8.pt"),
)


def _versions() -> dict:
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _quantize_inplace(model):
    """int8 dynamic quantization of Linear layers without copying the model."""
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    gc.collect()
    return model


def _load_fp32_and_quantize(model_name: str, device: torch.device):
    from transformers import BertModel
    bert = BertModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    bert.eval()
    if device.type == "cpu":
        logger.info("Applying 8-bit dynamic quantization to InLegalBERT (in place)...")
        bert = _quantize_inplace(bert)
    else:
        bert = bert.to(device)
    return bert


def load_bert_low_mem(model_name: str, device: torch.device,
                      quantized_path: str = QUANTIZED_BERT_PATH):
    """
    Return (tokenizer, bert) for inference.

    On CPU, loads the prebuilt int8 model if it exists and matches the installed
    torch / transformers versions; otherwise quantizes in place at startup.
    """
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(model_name)

    if device.type == "cpu" and os.path.isfile(quantized_path):
        try:
            # The file is produced by build_quantized_bert() inside our own image.
            payload = torch.load(quantized_path, map_location="cpu", weights_only=False)
            if payload.get("versions") == _versions() and payload.get("model_name") == model_name:
                bert = payload["model"]
                bert.eval()
                del payload
                gc.collect()
                logger.info("Loaded prebuilt int8 InLegalBERT from %s", quantized_path)
                return tok, bert
            logger.warning(
                "Prebuilt InLegalBERT at %s was built with %s (installed: %s); rebuilding in memory.",
                quantized_path, payload.get("versions"), _versions(),
            )
            del payload
            gc.collect()
        except Exception as e:  # corrupt / incompatible file -> fall back
            logger.warning("Could not load prebuilt InLegalBERT (%s); quantizing at startup.", e)

    logger.info("Loading InLegalBERT (%s)...", model_name)
    return tok, _load_fp32_and_quantize(model_name, device)


def build_quantized_bert(model_name: str = "law-ai/InLegalBERT",
                         out_path: str = QUANTIZED_BERT_PATH) -> str:
    """Build-time step: save an int8 InLegalBERT so the server never loads fp32."""
    from transformers import BertTokenizer
    BertTokenizer.from_pretrained(model_name)          # cache the tokenizer files too
    bert = _load_fp32_and_quantize(model_name, torch.device("cpu"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": bert, "model_name": model_name, "versions": _versions()}, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved int8 InLegalBERT to {out_path} ({size_mb:.0f} MB)")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_quantized_bert()
