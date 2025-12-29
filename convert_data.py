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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Polarity mapping
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
                edges.append(word.head)
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
        return {
            "token": tokens,
            "postag": ["NOUN"] * len(tokens),
            "edges": list(range(len(tokens))),
            "deprels": ["dep"] * len(tokens)
        }


def create_scope(aspect_from: int, aspect_to: int, text_len: int, window: int = 3) -> list:
    """Create scope window around aspect."""
    left = max(0, aspect_from - window)
    right = min(text_len - 1, aspect_to + window)
    return [left, right]


def convert_sample(sample: dict, image_features: dict, nlp) -> dict:
    """Convert single JSON sample to DASCO pkl format."""
    
    # Get image features
    url = sample.get("photo_url", "")
    if url in image_features:
        image_feature = image_features[url]["features"]
    else:
        # Create dummy features if not found (should not happen normally)
        print(f"Warning: Image not found in cache: {url[:50]}...")
        image_feature = np.zeros((257, 1408), dtype=np.float32)
    
    # Scene graph (photo_caption)
    scene_graph = sample.get("photo_caption", "")
    
    # Text processing
    text = sample.get("review", "")
    text_tokens = text.split()
    text_len = len(text_tokens)
    
    # Dependency parsing
    parse_result = run_dependency_parsing(text, nlp)
    
    # Build aspects with polarity and scope
    aspects = []
    review_aspects = sample.get("review_aspects", [])
    opinion_categories = sample.get("review_opinion_categories", [])
    
    for i, aspect in enumerate(review_aspects):
        polarity = POLARITY_MAP.get(
            opinion_categories[i] if i < len(opinion_categories) else "Unknown",
            "NEU"
        )
        scope = create_scope(aspect["from"], aspect["to"], text_len)
        aspects.append({
            "term": aspect["term"],
            "from": aspect["from"],
            "to": aspect["to"],
            "polarity": polarity,
            "scope": scope
        })
    
    # Build nouns (use aspects as nouns)
    nouns = []
    for aspect in aspects:
        nouns.append({
            "term": aspect["term"],
            "from": aspect["from"],
            "to": aspect["to"],
            "scope": aspect["scope"]
        })
    
    # Query input and text input
    query_input = "Extract aspect terms and their sentiment polarity"
    target = [asp["term"] for asp in aspects] if aspects else [text]
    
    # Parse info for DASCO format
    parse_info = {
        **parse_result,
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
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_idx = i // batch_size
        output_path = os.path.join(output_dir, f"data_{batch_idx}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(batch, f)
    
    print(f"Saved {len(data)} samples to {output_dir} ({len(data) // batch_size + 1} files)")


def main(args):
    # Load image cache
    print(f"Loading image cache from {args.image_cache}...")
    with open(args.image_cache, 'rb') as f:
        image_features = pickle.load(f)
    print(f"Loaded {len(image_features)} cached images")
    
    # Initialize Stanza
    print("Initializing Stanza NLP pipeline...")
    try:
        stanza.download('en', verbose=False)
    except:
        pass
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)
    
    # Load input data
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Filter samples with valid images
    valid_data = [s for s in data if s.get("photo_url") in image_features]
    print(f"Valid samples (with cached images): {len(valid_data)}")
    
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
        print(f"\nConverting {split_name} set...")
        converted = []
        for sample in tqdm(split_data, desc=f"Converting {split_name}"):
            try:
                result = convert_sample(sample, image_features, nlp)
                converted.append(result)
            except Exception as e:
                print(f"Error converting sample: {e}")
                continue
        
        output_dir = os.path.join(args.output, split_name)
        save_as_pkl(converted, output_dir, args.batch_size)
    
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
