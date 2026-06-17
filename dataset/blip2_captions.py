"""BLIP2 caption generation for images.

This module provides functions to generate captions for images using BLIP2 model.
Captions are saved to CSV for later use.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from PIL import Image

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    # Try BLIP2 first, fallback to BLIP
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        BLIP2_AVAILABLE = True
        BLIP_AVAILABLE = True
    except ImportError:
        BLIP2_AVAILABLE = False
        BLIP_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    BLIP_AVAILABLE = False
    print("Warning: transformers library not available. BLIP/BLIP2 caption generation will be disabled.")


BATCH_SIZE = 16  # Batch size for caption generation


def _load_blip2_model(device: torch.device):
    """Load BLIP2/BLIP model and processor.
    
    Tries BLIP2 first, falls back to BLIP if BLIP2 is not available.
    Uses "Salesforce/blip-image-captioning-base" as the default model.
    
    Args:
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor, model_name)
    """
    if not BLIP2_AVAILABLE and not BLIP_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    print("Loading BLIP/BLIP2 model...")
    
    # Try BLIP2 first (if available)
    if BLIP2_AVAILABLE:
        model_names = [
            "Salesforce/blip2-opt-2.7b",  # BLIP2 with OPT-2.7B
        ]
        
        for model_name in model_names:
            try:
                print(f"Trying BLIP2 model: {model_name}")
                processor = Blip2Processor.from_pretrained(model_name)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                )
                model = model.to(device)
                model.eval()
                print(f"BLIP2 model loaded on {device}: {model_name}")
                return model, processor, model_name
            except Exception as e:
                print(f"Error loading BLIP2 {model_name}: {e}")
                continue
    
    # Fallback to BLIP (always available if transformers is installed)
    if BLIP_AVAILABLE:
        model_name = "Salesforce/blip-image-captioning-base"
        try:
            print(f"Using BLIP model: {model_name}")
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            model = model.to(device)
            model.eval()
            print(f"BLIP model loaded on {device}: {model_name}")
            return model, processor, model_name
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model {model_name}: {e}")
    
    raise RuntimeError("No BLIP/BLIP2 model available")


def generate_captions(
    model,
    processor,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    num_items: int,
) -> Dict[int, str]:
    """Generate captions for all images in meta.
    
    Args:
        model: BLIP/BLIP2 model
        processor: BLIP/BLIP2 processor
        device: Device to run inference on
        meta: Dict {item_id: {image_path: str, ...}}
        num_items: Total number of items
        
    Returns:
        Dict {item_id: caption} - Captions for items with valid images
    """
    if not BLIP2_AVAILABLE and not BLIP_AVAILABLE:
        return {}
    
    # Collect items with valid images
    items_with_img = []
    for item_id in range(1, num_items + 1):
        info = meta.get(item_id, {})
        image_path = info.get("image_path") or info.get("image")
        if image_path and os.path.isfile(image_path):
            items_with_img.append((item_id, image_path))
    
    if not items_with_img:
        print("No images found for caption generation")
        return {}
    
    print(f"Generating captions for {len(items_with_img)} images...")
    
    captions = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items_with_img), BATCH_SIZE), desc="BLIP2 captions"):
            batch = items_with_img[i : i + BATCH_SIZE]
            batch_images = []
            batch_ids = []
            
            for item_id, path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    batch_ids.append(item_id)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process images
                inputs = processor(images=batch_images, return_tensors="pt").to(device)
                
                # Generate captions
                generated_ids = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3,
                    do_sample=False,
                )
                
                # Decode captions
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                # Store captions
                for idx, item_id in enumerate(batch_ids):
                    if idx < len(generated_texts):
                        captions[item_id] = generated_texts[idx].strip()
                
            except Exception as e:
                print(f"Error generating captions for batch: {e}")
                continue
    
    print(f"Generated {len(captions)} captions")
    return captions


def maybe_generate_blip2_captions(
    dataset,
    data: Dict[str, Any],
    args,
) -> Optional[Dict[int, str]]:
    """Generate BLIP2 captions if enabled and images are available.
    
    Args:
        dataset: Dataset instance
        data: Dataset data dict
        args: Arguments with use_image and generate_caption flags
        
    Returns:
        Dict {item_id: caption} or None if not generated
    """
    if not hasattr(args, 'generate_caption') or not args.generate_caption:
        return None
    
    if not args.use_image:
        print("Warning: generate_caption requires --use_image flag")
        return None
    
    if not BLIP2_AVAILABLE and not BLIP_AVAILABLE:
        print("Warning: BLIP/BLIP2 not available. Skipping caption generation.")
        return None
    
    meta = data.get("meta", {})
    if not meta:
        print("Warning: No metadata found. Skipping caption generation.")
        return None
    
    num_items = max(meta.keys()) if meta else 0
    if num_items == 0:
        print("Warning: No items found. Skipping caption generation.")
        return None
    
    # Check if captions already exist
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    captions_path = preproc_folder / "blip2_captions.pt"
    
    if captions_path.exists():
        print(f"Loading existing captions from {captions_path}")
        captions = torch.load(captions_path, map_location="cpu")
        return captions
    
    # Generate captions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, processor, model_name = _load_blip2_model(device)
    captions = generate_captions(model, processor, device, meta, num_items)
    
    # Save captions to .pt file for caching (optional, CSV is the main storage)
    if captions:
        captions_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captions, captions_path)
        print(f"Saved captions cache to {captions_path}")
        print(f"Note: Captions will also be saved to CSV in dataset_single_export.csv")
    
    return captions

