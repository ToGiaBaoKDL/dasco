#!/usr/bin/env python3
"""
Cache BLIP-2 image features from URLs for DASCO training.
Processes images from text_image_dataset.json without downloading to disk.
"""

import json
import pickle
import os
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import Blip2Processor, Blip2Model
import argparse
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """Load image from URL without saving to disk."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None


def extract_blip2_features(image: Image.Image, processor, model, device) -> np.ndarray:
    """Extract BLIP-2 ViT features [257, 1408] or [257, 1664] depending on model."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        # Get vision encoder outputs
        vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
        image_embeds = vision_outputs.last_hidden_state  # [1, 257, hidden_size]
    return image_embeds.cpu().numpy().squeeze()  # [257, hidden_size]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BLIP-2 model
    print("Loading BLIP-2 model (this may take a few minutes)...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print("BLIP-2 model loaded!")
    
    # Load input data
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get unique URLs
    url_to_id = {}
    for sample in data:
        url = sample.get("photo_url")
        if url and url not in url_to_id:
            url_to_id[url] = sample.get("image_id") or sample.get("id")
    
    print(f"Found {len(url_to_id)} unique images to process")
    
    # Extract features
    features_cache = {}
    failed_urls = []
    
    for url, img_id in tqdm(url_to_id.items(), desc="Extracting features"):
        image = load_image_from_url(url)
        if image is None:
            failed_urls.append(url)
            continue
        
        try:
            features = extract_blip2_features(image, processor, model, device)
            features_cache[url] = {
                "image_id": img_id,
                "features": features,
                "shape": features.shape
            }
        except Exception as e:
            print(f"Error extracting features for {url}: {e}")
            failed_urls.append(url)
    
    # Save cache
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(features_cache, f)
    
    print(f"\n=== Summary ===")
    print(f"Total images: {len(url_to_id)}")
    print(f"Successfully cached: {len(features_cache)}")
    print(f"Failed: {len(failed_urls)}")
    print(f"Feature shape: {list(features_cache.values())[0]['shape'] if features_cache else 'N/A'}")
    print(f"Cache saved to: {args.output}")
    
    if failed_urls:
        with open(args.output.replace('.pkl', '_failed.txt'), 'w') as f:
            f.write('\n'.join(failed_urls))
        print(f"Failed URLs saved to: {args.output.replace('.pkl', '_failed.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache BLIP-2 image features")
    parser.add_argument("--input", type=str, default="./data/text_image_dataset.json",
                        help="Input JSON file")
    parser.add_argument("--output", type=str, default="./image_cache/features.pkl",
                        help="Output cache file")
    args = parser.parse_args()
    main(args)
