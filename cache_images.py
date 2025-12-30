#!/usr/bin/env python3
"""
Cache BLIP-2 image features from URLs for DASCO training.
Optimized version with GPU, batch processing and parallel download.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


def load_image_from_url(url: str, timeout: int = 30) -> tuple:
    """Load image from URL without saving to disk. Returns (url, image or None)."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return (url, image)
    except Exception as e:
        return (url, None)


def download_images_parallel(urls: list, max_workers: int = 16) -> dict:
    """Download multiple images in parallel using ThreadPoolExecutor."""
    url_to_image = {}
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_image_from_url, url): url for url in urls}
        
        pbar = tqdm(as_completed(futures), total=len(urls), desc="Downloading images", unit="img")
        for future in pbar:
            url, image = future.result()
            if image is not None:
                url_to_image[url] = image
            else:
                failed.append(url)
            pbar.set_postfix({"ok": len(url_to_image), "failed": len(failed)})
    
    return url_to_image, failed


def extract_features_batch(images: list, processor, model, device, batch_size: int = 8) -> list:
    """Extract BLIP-2 features for a batch of images."""
    all_features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
            image_embeds = vision_outputs.last_hidden_state
        
        # Convert to numpy and add to list
        for j in range(image_embeds.shape[0]):
            all_features.append(image_embeds[j].cpu().numpy())
    
    return all_features


def main(args):
    # Force GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠ Warning: CUDA not available, using CPU (will be slow)")
    
    # Load BLIP-2 model
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print("✓ BLIP-2 model loaded!")
    
    # Load input data
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    
    # Get unique URLs
    print("Collecting unique image URLs...")
    url_to_id = {}
    for sample in data:
        url = sample.get("photo_url")
        if url and url not in url_to_id:
            url_to_id[url] = sample.get("image_id") or sample.get("id")
    print(f"✓ Found {len(url_to_id)} unique images")
    
    # Step 1: Download all images in parallel
    print(f"\n{'='*50}")
    print(f"Step 1: Downloading images (parallel, {args.workers} workers)...")
    urls = list(url_to_id.keys())
    url_to_image, failed_urls = download_images_parallel(urls, max_workers=args.workers)
    print(f"✓ Downloaded: {len(url_to_image)}, Failed: {len(failed_urls)}")
    
    # Step 2: Extract features in batches
    print(f"\n{'='*50}")
    print(f"Step 2: Extracting features (batch_size={args.batch_size})...")
    
    features_cache = {}
    urls_with_images = list(url_to_image.keys())
    images_list = [url_to_image[url] for url in urls_with_images]
    
    # Process in batches
    pbar = tqdm(range(0, len(images_list), args.batch_size), desc="Extracting features", unit="batch")
    for i in pbar:
        batch_urls = urls_with_images[i:i + args.batch_size]
        batch_images = images_list[i:i + args.batch_size]
        
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
                image_embeds = vision_outputs.last_hidden_state
            
            for j, url in enumerate(batch_urls):
                features_cache[url] = {
                    "image_id": url_to_id[url],
                    "features": image_embeds[j].cpu().numpy(),
                    "shape": image_embeds[j].shape
                }
                
            pbar.set_postfix({"cached": len(features_cache)})
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
            # Fallback to individual processing
            for url, img in zip(batch_urls, batch_images):
                try:
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
                        image_embeds = vision_outputs.last_hidden_state
                    features_cache[url] = {
                        "image_id": url_to_id[url],
                        "features": image_embeds[0].cpu().numpy(),
                        "shape": image_embeds[0].shape
                    }
                except:
                    failed_urls.append(url)
    
    # Clear GPU memory
    del images_list, url_to_image
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Save cache
    print(f"\n{'='*50}")
    print("Saving cache...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(features_cache, f)
    
    print(f"\n{'='*50}")
    print("=== Summary ===")
    print(f"Total images: {len(url_to_id)}")
    print(f"Successfully cached: {len(features_cache)}")
    print(f"Failed: {len(failed_urls)}")
    print(f"Feature shape: {list(features_cache.values())[0]['shape'] if features_cache else 'N/A'}")
    print(f"Cache saved to: {args.output}")
    
    if failed_urls:
        failed_path = args.output.replace('.pkl', '_failed.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_urls))
        print(f"Failed URLs saved to: {failed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache BLIP-2 image features (optimized)")
    parser.add_argument("--input", type=str, default="./data/text_image_dataset.json",
                        help="Input JSON file")
    parser.add_argument("--output", type=str, default="./image_cache/features.pkl",
                        help="Output cache file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for feature extraction (adjust based on GPU memory)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel download workers")
    args = parser.parse_args()
    main(args)
