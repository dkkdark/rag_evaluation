from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_CROSS_ENCODER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
DEFAULT_GTE_RERANKER_REVISION = "a6258e9d2b1a11aa7bccdff9efde562bbca4393d"
_CROSS_ENCODER_CACHE: Dict[str, object] = {}


def _alibaba_model_kwargs(model_name: str) -> Dict[str, object]:
    if model_name.startswith("Alibaba-NLP/gte-"):
        return {
            "trust_remote_code": True,
            "revision": DEFAULT_GTE_RERANKER_REVISION,
            "local_files_only": True,
        }
    return {}


def _local_hf_snapshot_path(model_name: str, revision: str | None = None) -> str:
    cache_root = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    model_cache = cache_root / ("models--" + model_name.replace("/", "--"))
    if not model_cache.exists():
        return model_name

    snapshots_root = model_cache / "snapshots"
    candidate_paths = []
    if revision:
        candidate_paths.append(snapshots_root / revision)
    if snapshots_root.exists():
        candidate_paths.extend(path for path in snapshots_root.iterdir() if path.is_dir())

    for path in candidate_paths:
        if (path / "config.json").exists() and (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
            or list(path.glob("*.safetensors"))
        ):
            return str(path)
    return model_name


def get_cross_encoder(model_name: str):
    cached_model = _CROSS_ENCODER_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if model_name.startswith("Alibaba-NLP/gte-"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_kwargs = _alibaba_model_kwargs(model_name)
        model_source = _local_hf_snapshot_path(model_name, DEFAULT_GTE_RERANKER_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(model_source, **model_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(model_source, **model_kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        cached_model = {
            "backend": "transformers_sequence_classification",
            "model_name": model_name,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
    else:
        from sentence_transformers import CrossEncoder

        cached_model = {
            "backend": "sentence_transformers_cross_encoder",
            "model_name": model_name,
            "model": CrossEncoder(model_name),
        }

    _CROSS_ENCODER_CACHE[model_name] = cached_model
    return cached_model


def _candidate_passage(row: Dict[str, object]) -> str:
    parts = [str(row.get("title") or ""), str(row.get("text") or "")]
    return "\n".join(part for part in parts if part.strip())


def _predict_scores(model_bundle: Dict[str, object], pairs: Sequence[tuple[str, str]]) -> List[float]:
    backend = str(model_bundle.get("backend") or "")
    if backend == "transformers_sequence_classification":
        import torch

        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]
        device = str(model_bundle.get("device") or "cpu")
        features = tokenizer(
            [query for query, _ in pairs],
            [passage for _, passage in pairs],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        features = {key: value.to(device) for key, value in features.items()}
        with torch.no_grad():
            logits = model(**features).logits
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits[:, 0]
        return [float(score) for score in logits.detach().cpu().tolist()]

    model = model_bundle["model"]
    return [float(score) for score in model.predict(list(pairs))]


def rerank_with_cross_encoder(
    *,
    query: str,
    rows: Sequence[Dict[str, object]],
    top_k: int,
    rerank_top_n: int = 10,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    fusion_weight: float = 0.65,
) -> List[Dict[str, object]]:
    if top_k <= 0 or not rows:
        return []

    limited_top_n = min(len(rows), max(top_k, rerank_top_n))
    head = [dict(row) for row in rows[:limited_top_n]]
    pairs = [(query, _candidate_passage(row)) for row in head]
    model_bundle = get_cross_encoder(model_name)
    scores = _predict_scores(model_bundle, pairs)

    base_scores = [float(row.get("score") or 0.0) for row in head]
    base_low = min(base_scores) if base_scores else 0.0
    base_high = max(base_scores) if base_scores else 0.0
    ce_low = min(scores) if scores else 0.0
    ce_high = max(scores) if scores else 0.0

    def normalize(value: float, low: float, high: float) -> float:
        if abs(high - low) < 1e-12:
            return 1.0 if high > 0 else 0.0
        return (value - low) / (high - low)

    reranked: List[Dict[str, object]] = []
    for row, ce_score in zip(head, scores):
        base_score = float(row.get("score") or 0.0)
        base_norm = normalize(base_score, base_low, base_high)
        ce_norm = normalize(float(ce_score), ce_low, ce_high)
        fused_score = ((1.0 - fusion_weight) * base_norm) + (fusion_weight * ce_norm)
        row["base_score_before_cross_encoder"] = base_score
        row["base_score_before_cross_encoder_normalized"] = base_norm
        row["cross_encoder_score"] = float(ce_score)
        row["cross_encoder_score_normalized"] = ce_norm
        row["cross_encoder_fusion_weight"] = float(fusion_weight)
        row["score"] = float(fused_score)
        row["reranked"] = True
        row["reranker"] = "cross_encoder"
        reranked.append(row)

    reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return reranked[:top_k]
