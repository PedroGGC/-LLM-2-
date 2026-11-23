"""Generic multimodal data contract supporting HuggingFace streaming (FineVision, PixMo, curated-danbooru-2026) and local JSONL."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
from typing import Iterator, Sequence


REFUSAL_PATTERNS = (
    "i'm sorry, i cannot",
    "i am sorry, i cannot",
    "i cannot assist with",
    "i can't assist with",
    "as an ai language model",
    "as an ai, i cannot",
)

_LORA_TAG_RE = re.compile(r"<(?:lora|lyco|hypernet|embedding):[^>]+>", re.IGNORECASE)
_CLIP_WEIGHT_RE = re.compile(r"\(([^():]+):[0-9.]+\)")
_CIVITAI_SPAM = {
    "masterpiece",
    "best quality",
    "high quality",
    "ultra-detailed",
    "ultra detailed",
    "extremely detailed",
    "absurdres",
    "highres",
    "newest",
    "latest",
    "8k",
    "4k",
    "uhd",
    "raw photo",
    "photorealistic",
    "score_9",
    "score_8_up",
    "score_7_up",
    "score_6_up",
    "score_5_up",
    "score_4_up",
    "source_anime",
    "rating_explicit",
}

# Verified open HF presets for the 4-part Lumina V1 mixture (60% general, 20% art, 10% CivitAI 3D/realism, 10% text SFT).
DEFAULT_VISION_PRESETS: dict[str, dict] = {
    "finevision_general": {
        "repo": "HuggingFaceM4/FineVision",
        "config": "allava_laion",
        "split": "train",
        "role": "general_vision",
        "weight": 0.45,
        "description": "General multimodal pretraining/VQA/OCR/captioning corpus.",
    },
    "pixmo_cap_en": {
        "repo": "VillanovaAI/multi-pixmo-cap",
        "config": "en",
        "split": "train",
        "role": "general_vision",
        "weight": 0.15,
        "description": "Dense descriptive natural-language image captions.",
    },
    "curated_danbooru_2026": {
        "repo": "aipracticecafe/curated-danbooru-2026",
        "split": "train",
        "role": "art_specialization",
        "weight": 0.20,
        "description": "330k curated anime illustrations (AVIF in Parquet) with natural-language upsampled captions.",
    },
    "civitai_nsfw_3d": {
        "repo": "wallstoneai/civitai-top-nsfw-images-with-metadata",
        "split": "train",
        "role": "realism_3d_adult",
        "weight": 0.10,
        "sanitize_civitai": True,
        "description": "Top-rated CivitAI realism/3D/mixed-style adult images with auto-cleaned prompts.",
    },
}

DEFAULT_TEXT_SFT_PRESETS: dict[str, dict] = {
    "booru_to_nl": {
        "repo": "Mexes/booru-tags-to-natural-language",
        "split": "train",
        "role": "tag_to_nl_text",
        "description": "Translates raw booru tag lists into natural descriptive prose.",
    },
    "uncensor_ptbr": {
        "repo": "Guilherme34/uncensor-ptbr",
        "split": "train",
        "role": "uncensored_sft_ptbr",
        "description": "Brazilian Portuguese uncensored instruction-following SFT.",
    },
    "uncensored_sft_v2": {
        "repo": "Pradip24042020/Uncensored-SFT-v2",
        "split": "train",
        "role": "uncensored_sft_en",
        "description": "General English uncensored instruction-following SFT.",
    },
    "mini_orca_code_uncensored": {
        "repo": "Vezora/Mini_Orca_Code_Uncensored_alpaca_Format",
        "split": "train",
        "role": "uncensored_code_sft",
        "description": "Uncensored code generation instruction-following SFT.",
    },
}


def clean_civitai_prompt(text: str) -> str:
    """Strip SD/CivitAI LoRA tags, CLIP weights (word:1.3), and quality-tag spam into clean description."""
    s = _LORA_TAG_RE.sub("", str(text or ""))
    # Iteratively unwrap nested CLIP weight parens (word:1.25) -> word
    for _ in range(3):
        nxt = _CLIP_WEIGHT_RE.sub(r"\1", s)
        if nxt == s:
            break
        s = nxt
    s = s.replace("BREAK", ",").replace("\n", ", ")
    parts = []
    seen = set()
    for raw_tok in s.split(","):
        tok = raw_tok.strip(" ()[]{}\t\r\n")
        low = tok.lower()
        if not tok or low in _CIVITAI_SPAM or low in seen:
            continue
        seen.add(low)
        parts.append(tok)
    return ", ".join(parts)


@dataclass
class MultimodalSample:
    images: list  # list[PIL.Image.Image]
    prompt: str
    response: str
    source: str
    rating: str | None = None


def is_refusal_text(text: str) -> bool:
    low = text.strip().lower()
    return any(pat in low for pat in REFUSAL_PATTERNS)


def _to_pil_image(obj):
    try:
        import pillow_avif  # noqa: F401
    except ImportError:
        pass
    from PIL import Image

    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            return Image.open(obj["path"]).convert("RGB")
    if isinstance(obj, (str, Path)):
        return Image.open(obj).convert("RGB")
    raise ValueError(f"Unsupported image object type: {type(obj).__name__}")


def _extract_images(row: dict) -> list:
    for key in ("images", "image", "jpg", "png", "webp", "avif", "img"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            imgs = [_to_pil_image(item) for item in val if item is not None]
            if imgs:
                return imgs
        else:
            return [_to_pil_image(val)]
    return []


def _extract_qa_pair(row: dict, default_prompt: str = "Describe this image in detail.") -> tuple[str, str] | None:
    # 1. Conversational formats (FineVision / LLaVA / ShareGPT / dataset_anima_vlm.jsonl)
    for conv_key in ("texts", "messages", "conversations"):
        conv = row.get(conv_key)
        if isinstance(conv, list) and conv:
            first = conv[0]
            if isinstance(first, dict):
                if "user" in first and "assistant" in first:
                    u = str(first["user"]).strip()
                    a = str(first["assistant"]).strip()
                    if a:
                        return u or default_prompt, a
                user_msg = None
                asst_msg = None
                for turn in conv:
                    role = str(turn.get("role") or turn.get("from") or "").lower()
                    content = turn.get("content") if "content" in turn else turn.get("value", "")
                    if isinstance(content, list):
                        parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        content = " ".join(p.strip() for p in parts if p.strip())
                    text = str(content).strip()
                    if role in ("user", "human") and user_msg is None:
                        user_msg = text
                    elif role in ("assistant", "gpt", "model") and asst_msg is None:
                        asst_msg = text
                if asst_msg:
                    return (user_msg or default_prompt), asst_msg

    # 2. Top-level user/assistant (+ optional system, e.g. AnimaDirector dataset_anima_vlm.jsonl) & prompt/response pairs
    if row.get("assistant") is not None and str(row.get("assistant")).strip():
        resp = str(row["assistant"]).strip()
        user_txt = str(row.get("user") or row.get("prompt") or default_prompt).strip()
        sys_txt = str(row.get("system") or "").strip()
        prompt = f"{sys_txt}\n\n{user_txt}" if sys_txt else user_txt
        return prompt, resp

    if row.get("response") or row.get("answer") or row.get("output"):
        resp = str(row.get("response") or row.get("answer") or row.get("output")).strip()
        instr = str(row.get("instruction") or row.get("prompt") or row.get("question") or default_prompt).strip()
        inp = str(row.get("input") or "").strip()
        prompt = f"{instr}\n{inp}" if inp else instr
        if resp:
            return prompt, resp

    # 2b. CivitAI metadata dict/JSON containing nested "prompt"
    meta = row.get("meta") or row.get("metadata")
    if isinstance(meta, str) and meta.strip().startswith("{"):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = None
    if isinstance(meta, dict):
        mp = meta.get("prompt") or meta.get("Prompt")
        if mp is not None and str(mp).strip():
            return default_prompt, str(mp).strip()

    # 3. Caption / upsampled prompt columns (PixMo / curated-danbooru-2026)
    for cap_key in (
        "caption_long",
        "refined_caption",
        "upsampled_caption",
        "caption",
        "text",
        "description",
        "prompt_upsampled",
        "prompt",
        "tag_string",
        "tags",
    ):
        val = row.get(cap_key)
        if val is not None and str(val).strip():
            return default_prompt, str(val).strip()
    return None


def iter_jsonl_multimodal(
    jsonl_path: str | Path,
    *,
    image_root: str | Path | None = None,
    source_name: str = "local_jsonl",
    filter_refusals: bool = True,
    max_samples: int | None = None,
) -> Iterator[MultimodalSample]:
    """Stream MultimodalSample rows from a local JSONL file (compatible with AnimaDirector & standard VLM JSONL)."""
    path = Path(jsonl_path)
    root = Path(image_root) if image_root is not None else path.parent
    yielded = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            img_refs = row.get("images")
            if img_refs is None and row.get("image") is not None:
                img_refs = [row["image"]]
            if not img_refs:
                continue
            resolved = []
            for ref in img_refs:
                p = Path(ref)
                if not p.is_absolute():
                    p = root / p
                resolved.append(_to_pil_image(p))
            qa = _extract_qa_pair(row)
            if qa is None:
                continue
            prompt, response = qa
            if filter_refusals and is_refusal_text(response):
                continue
            yield MultimodalSample(
                images=resolved,
                prompt=prompt,
                response=response,
                source=source_name,
                rating=None if row.get("rating") is None else str(row["rating"]),
            )
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def iter_hf_multimodal(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    max_samples: int = 256,
    max_rows_scanned: int | None = None,
    filter_refusals: bool = True,
    sanitize_civitai: bool = False,
    allowed_ratings: Sequence[str] | None = None,
    streaming: bool = True,
) -> Iterator[MultimodalSample]:
    """Stream MultimodalSample rows from any HuggingFace multimodal dataset."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, config_name, split=split, streaming=streaming) if config_name else load_dataset(repo_id, split=split, streaming=streaming)
    allowed = None if allowed_ratings is None else {r.lower() for r in allowed_ratings}
    scan_limit = max(max_samples * 10, 500) if max_rows_scanned is None else int(max_rows_scanned)
    yielded = 0
    scanned = 0
    for row in ds:
        if yielded >= max_samples or scanned >= scan_limit:
            return
        scanned += 1
        if not isinstance(row, dict):
            continue
        rating = row.get("rating")
        if allowed is not None and rating is not None and str(rating).lower() not in allowed:
            continue
        try:
            imgs = _extract_images(row)
        except Exception:
            continue
        if not imgs:
            continue
        qa = _extract_qa_pair(row)
        if qa is None:
            continue
        prompt, response = qa
        if sanitize_civitai:
            response = clean_civitai_prompt(response)
            if len(response) < 8:
                continue
        if filter_refusals and is_refusal_text(response):
            continue
        yield MultimodalSample(
            images=imgs,
            prompt=prompt,
            response=response,
            source=repo_id,
            rating=None if rating is None else str(rating),
        )
        yielded += 1


def iter_hf_text_sft(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    max_samples: int = 256,
    max_rows_scanned: int | None = None,
    filter_refusals: bool = True,
    streaming: bool = True,
) -> Iterator[MultimodalSample]:
    """Stream text-only uncensored SFT / tag-to-NL rows as MultimodalSample(images=[]) for Stage 2/3 mixing."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, config_name, split=split, streaming=streaming) if config_name else load_dataset(repo_id, split=split, streaming=streaming)
    scan_limit = max(max_samples * 10, 500) if max_rows_scanned is None else int(max_rows_scanned)
    yielded = 0
    scanned = 0
    for row in ds:
        if yielded >= max_samples or scanned >= scan_limit:
            return
        scanned += 1
        if not isinstance(row, dict):
            continue
        qa = _extract_qa_pair(row, default_prompt="Follow the user instruction accurately.")
        if qa is None:
            continue
        prompt, response = qa
        if len(response) < 4:
            continue
        if filter_refusals and is_refusal_text(response):
            continue
        yield MultimodalSample(images=[], prompt=prompt, response=response, source=repo_id, rating="text_sft")
        yielded += 1


def mix_multimodal_streams(
    streams: Sequence[Iterator[MultimodalSample]],
    weights: Sequence[float] | None = None,
    *,
    max_samples: int | None = None,
) -> Iterator[MultimodalSample]:
    """Deterministic weighted round-robin mixer across general + specialized multimodal streams."""
    if not streams:
        return
    if weights is None:
        w = [1.0] * len(streams)
    else:
        if len(weights) != len(streams) or any(x <= 0 for x in weights):
            raise ValueError("weights must match streams length and be positive")
        w = list(weights)
    active = [True] * len(streams)
    credits = [0.0] * len(streams)
    yielded = 0
    while any(active):
        best_idx = -1
        best_score = -1.0
        for idx, is_alive in enumerate(active):
            if not is_alive:
                continue
            credits[idx] += w[idx]
            if credits[idx] > best_score:
                best_score = credits[idx]
                best_idx = idx
        if best_idx < 0:
            return
        credits[best_idx] -= sum(w[i] for i, a in enumerate(active) if a)
        try:
            sample = next(streams[best_idx])
        except StopIteration:
            active[best_idx] = False
            continue
        yield sample
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            return
