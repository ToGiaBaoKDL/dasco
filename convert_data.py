#!/usr/bin/env python3
"""
Convert JSON data to DASCO pkl format for training.
Splits data into train/dev/test and converts to required format.
"""

import json
import pickle
import os
import numpy as np
import stanza
import argparse
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Polarity mapping - must match ParseData expected format
POLARITY_MAP = {
    "Positive": "POS",
    "Negative": "NEG",
    "Neutral": "NEU",
    "Unknown": "NEU"
}


def run_dependency_parsing(text: str, nlp) -> dict:
    """Run Stanza dependency parsing."""
    try:
        doc = nlp(text)
        tokens = []
        postag = []
        edges = []
        deprels = []
        
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                postag.append(word.upos)
                edges.append(word.head)  # head index (1-indexed, 0 for root)
                deprels.append(word.deprel)
        
        return {
            "token": tokens,
            "postag": postag,
            "edges": edges,
            "deprels": deprels
        }
    except Exception as e:
        print(f"Parsing error: {e}")
        # Fallback: simple tokenization
        tokens = text.split()
        n = len(tokens)
        return {
            "token": tokens,
            "postag": ["NOUN"] * n,
            "edges": [0] + list(range(1, n)),  # Simple chain dependency
            "deprels": ["root"] + ["dep"] * (n - 1) if n > 0 else []
        }


def create_scope_offsets(aspect_from: int, aspect_to: int, text_len: int, window: int = 3) -> list:
    """
    Create scope as [left_offset, right_offset].
    ParseData expects: aspect_scope = [id_b - s_b, s_e - id_e]
    So we need to return [start_pos, end_pos] where:
    - start_pos = aspect_from - left_offset
    - end_pos = aspect_to + right_offset
    """
    left_offset = min(window, aspect_from)
    right_offset = min(window, text_len - 1 - aspect_to) if text_len > 0 else 0
    # scope format: [start_pos, end_pos] of the scope window
    start_pos = aspect_from - left_offset
    end_pos = aspect_to + right_offset
    return [start_pos, end_pos]


def convert_sample(sample: dict, image_features: dict, nlp) -> dict:
    """Convert single JSON sample to DASCO pkl format."""
    
    # Get image features
    url = sample.get("photo_url", "")
    if url in image_features:
        image_feature = image_features[url]["features"]
    else:
        print(f"Warning: Image not found in cache: {url[:50]}...")
        image_feature = np.zeros((257, 1408), dtype=np.float32)
    
    # Scene graph (photo_caption)
    scene_graph = sample.get("photo_caption", "")
    
    # Text processing
    text = sample.get("review", "")
    
    # Dependency parsing - MUST run before building aspects
    parse_result = run_dependency_parsing(text, nlp)
    text_tokens = parse_result["token"]  # Use parsed tokens, not simple split
    text_len = len(text_tokens)
    
    # Build aspects with polarity and scope
    aspects = []
    review_aspects = sample.get("review_aspects", [])
    opinion_categories = sample.get("review_opinion_categories", [])
    
    for i, aspect in enumerate(review_aspects):
        polarity = POLARITY_MAP.get(
            opinion_categories[i] if i < len(opinion_categories) else "Unknown",
            "NEU"
        )
        
        # Validate aspect positions
        aspect_from = aspect["from"]
        aspect_to = aspect["to"]
        
        # Clamp to valid range
        aspect_from = max(0, min(aspect_from, text_len - 1)) if text_len > 0 else 0
        aspect_to = max(0, min(aspect_to, text_len - 1)) if text_len > 0 else 0
        
        # Create scope window
        scope = create_scope_offsets(aspect_from, aspect_to, text_len)
        
        aspects.append({
            "term": aspect["term"],
            "from": aspect_from,
            "to": aspect_to,
            "polarity": polarity,
            "scope": scope
        })
    
    # Build nouns from aspects (each aspect term is also a noun)
    nouns = []
    for aspect in aspects:
        nouns.append({
            "term": aspect["term"],
            "from": aspect["from"],
            "to": aspect["to"],
            "scope": aspect["scope"]  # Same scope format
        })
    
    # Query input
    query_input = "Extract aspect terms and their sentiment polarity"
    
    # Target: list of aspect terms
    target = [asp["term"] for asp in aspects] if aspects else [text]
    
    # Parse info - critical structure for ParseData function
    parse_info = {
        "token": parse_result["token"],
        "postag": parse_result["postag"],
        "edges": parse_result["edges"],
        "deprels": parse_result["deprels"],
        "aspects": [{
            "term": asp["term"],
            "from": asp["from"],
            "to": asp["to"],
            "polarity": asp["polarity"],
            "scope": asp["scope"]
        } for asp in aspects]
    }
    
    return {
        "image_feature": image_feature,
        "query_input": query_input,
        "scene_graph": scene_graph,
        "text_input": text,
        "target": target,
        "nouns": nouns,
        "parse_info": parse_info
    }


def save_as_pkl(data: list, output_dir: str, batch_size: int = 100):
    """Save data as pkl files in batches."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_idx = i // batch_size
        output_path = os.path.join(output_dir, f"data_{batch_idx}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(batch, f)
    
    print(f"Saved {len(data)} samples to {output_dir} ({num_batches} files)")


def main(args):
    # Load image cache
    print(f"Loading image cache from {args.image_cache}...")
    with open(args.image_cache, 'rb') as f:
        image_features = pickle.load(f)
    print(f"✓ Loaded {len(image_features)} cached images")
    
    # Initialize Stanza with GPU if available
    print("Initializing Stanza NLP pipeline...")
    use_gpu = torch.cuda.is_available()
    try:
        stanza.download('en', verbose=False)
    except:
        pass
    nlp = stanza.Pipeline(
        'en', 
        processors='tokenize,pos,lemma,depparse', 
        verbose=False,
        use_gpu=use_gpu
    )
    print(f"✓ Stanza initialized (GPU: {use_gpu})")
    
    # Load input data
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    
    # Filter samples with valid images
    print("Filtering samples with cached images...")
    valid_data = [s for s in tqdm(data, desc="Filtering") if s.get("photo_url") in image_features]
    print(f"✓ Valid samples: {len(valid_data)} / {len(data)}")
    
    if len(valid_data) == 0:
        print("ERROR: No valid samples found! Check image cache.")
        return
    
    # Split data
    train_data, temp_data = train_test_split(
        valid_data, 
        test_size=(args.val_ratio + args.test_ratio),
        random_state=42
    )
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_ratio_adjusted),
        random_state=42
    )
    
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Convert and save each split
    for split_name, split_data in [("train", train_data), ("dev", val_data), ("test", test_data)]:
        print(f"\n{'='*50}")
        print(f"Converting {split_name} set ({len(split_data)} samples)...")
        converted = []
        errors = 0
        pbar = tqdm(split_data, desc=f"Converting {split_name}", unit="sample")
        for sample in pbar:
            try:
                result = convert_sample(sample, image_features, nlp)
                converted.append(result)
                pbar.set_postfix({"converted": len(converted), "errors": errors})
            except Exception as e:
                errors += 1
                pbar.set_postfix({"converted": len(converted), "errors": errors})
                continue
        
        output_dir = os.path.join(args.output, split_name)
        save_as_pkl(converted, output_dir, args.batch_size)
        print(f"✓ {split_name}: {len(converted)} samples saved ({errors} errors)")
    
    print("\n=== Conversion Complete ===")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to DASCO pkl format")
    parser.add_argument("--input", type=str, default="./data/text_image_dataset.json",
                        help="Input JSON file")
    parser.add_argument("--image_cache", type=str, default="./image_cache/features.pkl",
                        help="Image features cache file")
    parser.add_argument("--output", type=str, default="./finetune_dataset/custom",
                        help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Samples per pkl file")
    args = parser.parse_args()
    main(args)
