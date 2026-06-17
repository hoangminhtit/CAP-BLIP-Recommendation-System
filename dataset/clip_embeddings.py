"""CLIP embedding extraction integrated with the dataset pipeline.

This module provides a helper `maybe_extract_clip_embeddings` that can be
called right after preprocessing / loading `dataset.pkl` to compute CLIP
embeddings for item images and/or texts and store them alongside the
preprocessed data.

Usage (from `data_prepare.py`):

    dataset = dataset_factory(args)
    data = dataset.load_dataset()
    maybe_extract_clip_embeddings(dataset, data, args)

This will, at most once per preprocessed folder, create a file:

    <preprocessed_folder>/clip_embeddings.pt

containing a dict:

    {
        "model_name": str,
        "image_embs": FloatTensor[num_items+1, D] or None,
        "text_embs":  FloatTensor[num_items+1, D] or None,
    }

Row 0 is padding; rows 1..num_items correspond to dense item ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import os

import torch
from PIL import Image
from tqdm import tqdm


CLIP_MODEL_NAME = "ViT-B-32"  # open_clip model name
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 128


def _load_clip_model(device: torch.device):
    """Load CLIP model + preprocess + tokenizer from open_clip.

    Imported lazily to avoid hard dependency on environments that don't need it.
    """

    try:
        import open_clip  # type: ignore
    except ImportError as e:
        raise ImportError(
            "open_clip_torch is required for CLIP embeddings. "
            "Please install it (e.g. `pip install open_clip_torch`)."
        ) from e

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def _extract_image_embeddings(
    model,
    preprocess,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    num_items: int,
) -> torch.Tensor | None:
    """Return FloatTensor [num_items+1, D] with CLIP image embeddings.

    Missing images will have all zeros.
    """

    items_with_img = []
    for item_id in range(1, num_items + 1):
        info = meta.get(item_id, {})
        image_path = info.get("image_path") or info.get("image")
        if image_path and os.path.isfile(image_path):
            items_with_img.append((item_id, image_path))

    if not items_with_img:
        return None

    embs = None
    with torch.no_grad():
        for i in tqdm(range(0, len(items_with_img), BATCH_SIZE), desc="CLIP image embeddings"):
            batch = items_with_img[i : i + BATCH_SIZE]
            pil_images = []
            batch_ids = []
            for item_id, path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                pil_images.append(preprocess(img))
                batch_ids.append(item_id)

            if not pil_images:
                continue

            pixel_batch = torch.stack(pil_images).to(device)
            features = model.encode_image(pixel_batch)
            features = features / features.norm(dim=-1, keepdim=True)

            if embs is None:
                D = features.size(1)
                embs = torch.zeros(num_items + 1, D, dtype=torch.float32)

            for idx, item_id in enumerate(batch_ids):
                embs[item_id] = features[idx].cpu()

    return embs


def _extract_text_embeddings(
    model,
    tokenizer,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    num_items: int,
) -> torch.Tensor | None:
    """Return FloatTensor [num_items+1, D] with CLIP text embeddings.

    Missing texts will have all zeros.
    """

    items_with_text = []
    for item_id in range(1, num_items + 1):
        info = meta.get(item_id, {})
        text = info.get("text")
        if text and isinstance(text, str) and text.strip():
            items_with_text.append((item_id, text.strip()))

    if not items_with_text:
        return None

    embs = None
    with torch.no_grad():
        for i in tqdm(range(0, len(items_with_text), BATCH_SIZE), desc="CLIP text embeddings"):
            batch = items_with_text[i : i + BATCH_SIZE]
            texts = [t for _, t in batch]
            batch_ids = [item_id for item_id, _ in batch]

            tokens = tokenizer(texts).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

            if embs is None:
                D = features.size(1)
                embs = torch.zeros(num_items + 1, D, dtype=torch.float32)

            for idx, item_id in enumerate(batch_ids):
                embs[item_id] = features[idx].cpu()

    return embs


def maybe_extract_clip_embeddings(dataset, data: Dict[str, Any], args) -> None:
    """Run CLIP embedding extraction once per preprocessed dataset.

    - Only runs if at least one of `args.use_image` or `args.use_text` is True.
    - Skips if `<preprocessed_folder>/clip_embeddings.pt` already exists.
    - Uses `data['meta']` and `data['smap']` to know items and metadata.
    """

    if not (getattr(args, "use_image", False) or getattr(args, "use_text", False)):
        return

    preprocessed_folder = Path(dataset._get_preprocessed_folder_path())
    out_path = preprocessed_folder / "clip_embeddings.pt"

    if out_path.exists():
        print(f"CLIP embeddings already exist at {out_path}, skip extraction.")
        return

    meta: Dict[int, Dict[str, Any]] = data["meta"]
    num_items = len(data["smap"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting CLIP embeddings on device: {device}")
    model, preprocess, tokenizer = _load_clip_model(device)

    image_embs = None
    text_embs = None

    if getattr(args, "use_image", False):
        image_embs = _extract_image_embeddings(model, preprocess, device, meta, num_items)

    if getattr(args, "use_text", False):
        text_embs = _extract_text_embeddings(model, tokenizer, device, meta, num_items)

    payload = {
        "model_name": f"open_clip:{CLIP_MODEL_NAME}-{CLIP_PRETRAINED}",
        "image_embs": image_embs,
        "text_embs": text_embs,
    }
    torch.save(payload, out_path)
    print(f"Saved CLIP embeddings to: {out_path}")
