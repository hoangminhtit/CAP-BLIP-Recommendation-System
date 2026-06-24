"""Item semantic summary generation with Qwen3-4B.

This module builds short, factual summaries from item text plus optional
image-derived signals (caption or VIU). Summaries are cached per dataset
split to avoid repeated generation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from tqdm import tqdm
import pandas as pd

from dataset.paths import get_preprocessed_csv_path, get_preprocessed_folder_path
from evaluation.utils import load_dataset_from_csv

DEFAULT_MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"

try:
    from unsloth import FastLanguageModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: unsloth FastLanguageModel not available. Item summary generation will be skipped.")


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_qwen3_text_model(
    device: torch.device,
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = 2048,
    use_quantization: bool = True,
    use_torch_compile: bool = False,
):
    """Load Qwen3-4B text model with optional 4-bit quantization."""
    if not QWEN_AVAILABLE:
        raise ImportError(
            "unsloth FastLanguageModel is required. Install with: pip install unsloth[colab-new]"
        )

    load_in_4bit = use_quantization and device.type == "cuda"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map={"": local_rank} if device.type == "cuda" else None,
    )

    # Ensure padding token exists for batching
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Enable inference optimizations when available
    try:
        model = FastLanguageModel.for_inference(model)
    except Exception:
        pass

    if use_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: torch.compile failed, continuing without compile: {exc}")

    return model, tokenizer


def _build_summary_prompt(
    text: Optional[str],
    caption: Optional[str],
    viu: Optional[str],
    language: str = "en",
) -> str:
    """Create a concise, instruction-based prompt for summarization."""
    blocks = []
    if text:
        blocks.append(f"- Description: {text.strip()}")
    if caption:
        blocks.append(f"- Image caption: {caption.strip()}")
    if viu:
        blocks.append(f"- VIU: {viu.strip()}")

    content = "\n".join(blocks) if blocks else "- No content provided."

    return (f"""You are a product summarization assistant.

Write exactly ONE neutral, factual sentence (maximum 60 words) that accurately combines the provided details.
Use only information that is explicitly stated in the input.
Do NOT infer, assume, or add attributes (e.g., compatibility, included items, materials, sizes) unless clearly specified.
If information is unclear or conflicting, omit it rather than guessing.

Output language: {language}

Details:
{content}

Summary:
""")


def _select_auxiliary_text(
    caption: Optional[str],
    viu: Optional[str],
    source: str,
) -> Dict[str, Optional[str]]:
    """Select which auxiliary field to include based on mode."""
    if source == "caption":
        return {"caption": caption, "viu": None}
    if source == "viu":
        return {"caption": None, "viu": viu}
    if source == "text_only":
        return {"caption": None, "viu": None}

    # auto: prefer caption, fall back to VIU
    if caption:
        return {"caption": caption, "viu": None}
    return {"caption": None, "viu": viu}


def generate_item_summaries(
    model,
    tokenizer,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    captions: Optional[Dict[int, str]] = None,
    vius: Optional[Dict[int, str]] = None,
    source: str = "auto",
    language: str = "en",
    batch_size: int = 1,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    max_seq_length: int = 2048,
) -> Dict[int, str]:
    """Generate summaries for all items with available content."""
    item_ids = sorted(meta.keys())
    results: Dict[int, str] = {}

    batch_size = 1
    for start in tqdm(range(0, len(item_ids), batch_size), desc="Qwen3 item summaries"):
        batch_ids = item_ids[start : start + batch_size]
        prompts = []
        kept_ids = []
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for item_id in batch_ids:
            info = meta.get(item_id, {})
            text = info.get("text")
            caption_val = captions.get(item_id) if captions else info.get("caption")
            viu_val = vius.get(item_id) if vius else info.get("viu")

            if not text and not caption_val and not viu_val:
                continue

            aux = _select_auxiliary_text(caption_val, viu_val, source)
            prompt = _build_summary_prompt(text, aux["caption"], aux["viu"], language=language)
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(formatted)
            kept_ids.append(item_id)

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        input_lengths = inputs.get("attention_mask")
        if input_lengths is None:
            input_lengths = torch.sum(inputs["input_ids"] != pad_id, dim=1)
        else:
            input_lengths = input_lengths.sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=pad_id,
            )

        for idx, item_id in enumerate(kept_ids):
            gen_tokens = outputs[idx, int(input_lengths[idx]) :]
            summary = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results[item_id] = summary

    return results


def maybe_generate_item_summaries(
    dataset,
    data: Dict[str, Any],
    args,
    captions: Optional[Dict[int, str]] = None,
    vius: Optional[Dict[int, str]] = None,
) -> Optional[Dict[int, str]]:
    """Generate item summaries if requested via args."""
    if not hasattr(args, "generate_item_summary") or not args.generate_item_summary:
        return None

    meta = data.get("meta", {})
    if not meta:
        print("[item_summary] No metadata found. Skipping summary generation.")
        return None

    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    summaries_path = preproc_folder / "item_summaries_qwen3_4b.pt"

    if summaries_path.exists():
        print(f"[item_summary] Loading cached summaries from {summaries_path}")
        return torch.load(summaries_path, map_location="cpu")

    device = _resolve_device()
    print(f"[item_summary] Using device: {device}")

    try:
        max_seq_length = getattr(args, "qwen_max_seq_length", 2048)
    except Exception:
        max_seq_length = 2048

    try:
        model, tokenizer = _load_qwen3_text_model(
            device=device,
            max_seq_length=max_seq_length,
            use_quantization=getattr(args, "use_quantization", True),
            use_torch_compile=getattr(args, "use_torch_compile", False),
        )
    except ImportError as exc:
        print(f"[item_summary] {exc}")
        return None

    summaries = generate_item_summaries(
        model=model,
        tokenizer=tokenizer,
        device=device,
        meta=meta,
        captions=captions,
        vius=vius,
        source=getattr(args, "summary_source", "auto"),
        language=getattr(args, "summary_language", "en"),
        batch_size=getattr(args, "summary_batch_size", 4),
        max_new_tokens=getattr(args, "summary_max_new_tokens", 64),
        temperature=getattr(args, "summary_temperature", 0.7),
        max_seq_length=max_seq_length,
    )

    if summaries:
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(summaries, summaries_path)
        print(f"[item_summary] Saved {len(summaries)} summaries to {summaries_path}")

    return summaries if summaries else None


def generate_item_summaries_from_csv(args) -> Optional[Dict[int, str]]:
    """Standalone task: load dataset_single_export.csv then write item summaries.

    Steps:
    1) Load preprocessed CSV (run data_prepare.py beforehand).
    2) Generate summaries using Qwen3-4B (text-only) with optional caption/VIU.
    3) Save cache .pt and update CSV with new column item_summary.
    """

    if not hasattr(args, "generate_item_summary") or not args.generate_item_summary:
        print("[item_summary] Flag --generate_item_summary not set; skipping.")
        return None

    # Locate preprocessed CSV
    csv_path = get_preprocessed_csv_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, getattr(args, 'data_path', None))
    if not csv_path.exists():
        print(f"[item_summary] CSV not found at {csv_path}. Run data_prepare.py first.")
        return None

    # Load dataset splits/meta directly from CSV (no preprocessing rerun)
    data = load_dataset_from_csv(args.dataset_code, args.min_rating, args.min_uc, args.min_sc)
    meta = data.get("meta", {})
    if not meta:
        print("[item_summary] No metadata found. Skipping summary generation.")
        return None

    preproc_folder = get_preprocessed_folder_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, getattr(args, 'data_path', None))
    summaries_path = preproc_folder / "item_summaries_qwen3_4b.pt"

    if summaries_path.exists():
        print(f"[item_summary] Loading cached summaries from {summaries_path}")
        summaries = torch.load(summaries_path, map_location="cpu")
    else:
        device = _resolve_device()
        print(f"[item_summary] Using device: {device}")

        try:
            max_seq_length = getattr(args, "qwen_max_seq_length", 2048)
        except Exception:
            max_seq_length = 2048

        try:
            model, tokenizer = _load_qwen3_text_model(
                device=device,
                max_seq_length=max_seq_length,
                use_quantization=getattr(args, "use_quantization", True),
                use_torch_compile=getattr(args, "use_torch_compile", False),
            )
        except ImportError as exc:
            print(f"[item_summary] {exc}")
            return None

        captions = {k: v.get("caption") for k, v in meta.items() if v.get("caption")}
        vius = {k: v.get("viu") for k, v in meta.items() if v.get("viu")}

        summaries = generate_item_summaries(
            model=model,
            tokenizer=tokenizer,
            device=device,
            meta=meta,
            captions=captions,
            vius=vius,
            source=getattr(args, "summary_source", "auto"),
            language=getattr(args, "summary_language", "en"),
            batch_size=getattr(args, "summary_batch_size", 4),
            max_new_tokens=getattr(args, "summary_max_new_tokens", 64),
            temperature=getattr(args, "summary_temperature", 0.7),
            max_seq_length=max_seq_length,
        )

        if summaries:
            summaries_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(summaries, summaries_path)
            print(f"[item_summary] Saved {len(summaries)} summaries to {summaries_path}")

    if not summaries:
        print("[item_summary] No summaries generated.")
        return None

    # Update CSV with item_summary column
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[item_summary] Failed to read CSV {csv_path}: {exc}")
        return summaries

    df["item_summary"] = df["item_new_id"].map(lambda x: summaries.get(int(x), ""))
    try:
        df.to_csv(csv_path, index=False)
        print(f"[item_summary] Updated CSV with item_summary at {csv_path}")
    except Exception as exc:
        print(f"[item_summary] Failed to write updated CSV: {exc}")

    return summaries
