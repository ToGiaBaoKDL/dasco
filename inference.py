#!/usr/bin/env python3
"""
DASCO Inference Script
Run MABSA inference on a single JSON file and output results as JSON.
"""

import torch
import json
import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from model import from_pretrained
from dataset import twitter_dataset, collate_fn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


POLARITY_LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}


def load_models(mate_path: str, masc_path: str, device: str, gcn_layers: int = 5):
    """Load MATE and MASC models."""
    # Load MATE
    mate_args = argparse.Namespace(
        task='MATE', hyper1=0.25, hyper2=0.25, hyper3=0.25, gcn_layers=gcn_layers
    )
    mate_model = from_pretrained(mate_path, mate_args).to(device).eval()
    
    # Load MASC
    masc_args = argparse.Namespace(
        task='MASC', hyper1=0.25, hyper2=0.25, hyper3=0.25, gcn_layers=gcn_layers
    )
    masc_model = from_pretrained(masc_path, masc_args).to(device).eval()
    
    return mate_model, masc_model


def prepare_single_sample(sample: dict, image_features: dict, tokenizers: dict) -> dict:
    """Prepare a single sample for inference."""
    IE_tokenizer, PQ_tokenizer = tokenizers['IE'], tokenizers['PQ']
    
    # Get image features
    url = sample.get("photo_url", "")
    if url in image_features:
        image_feature = torch.from_numpy(image_features[url]["features"])
    else:
        image_feature = torch.zeros((257, 1408))
    
    # Scene graph
    scene_graph = sample.get("photo_caption", "")
    
    # Text
    text = sample.get("review", "")
    
    # Query
    query_input = "Extract aspect terms and their sentiment polarity"
    
    # Tokenize
    query_inputs = PQ_tokenizer(
        query_input,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]
    
    scene_graph_inputs = PQ_tokenizer(
        scene_graph,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    IE_inputs = IE_tokenizer(
        text=query_input,
        text_pair=text,
        padding="max_length",
        truncation=True,
        max_length=480,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    # Create dummy adj_matrix
    adj_matrix = torch.eye(512)
    
    return {
        "image_embeds": image_feature.unsqueeze(0),
        "query_inputs": query_inputs,
        "scene_graph": {
            "input_ids": scene_graph_inputs["input_ids"],
            "attention_mask": scene_graph_inputs["attention_mask"]
        },
        "IE_inputs": {
            "input_ids": IE_inputs["input_ids"],
            "attention_mask": IE_inputs["attention_mask"]
        },
        "adj_matrix": adj_matrix.unsqueeze(0),
        "aspects_mask": [torch.zeros(512).unsqueeze(0)],
        "aspects_scope": [torch.zeros(512).unsqueeze(0)],
        "nouns_mask": [torch.zeros(512).unsqueeze(0)],
        "nouns_scope": [torch.zeros(512).unsqueeze(0)],
        "noun_targets": [torch.tensor([0])],
        "aspect_targets": [torch.tensor([0])],
        "original_text": text,
        "original_aspects": sample.get("review_aspects", [])
    }


def move_batch_to_device(batch: dict, device: str):
    """Move batch tensors to device."""
    batch["image_embeds"] = batch["image_embeds"].to(device)
    batch["query_inputs"] = batch["query_inputs"].to(device)
    batch["scene_graph"]['input_ids'] = batch["scene_graph"]['input_ids'].to(device)
    batch["scene_graph"]['attention_mask'] = batch["scene_graph"]['attention_mask'].to(device)
    batch["IE_inputs"]['input_ids'] = batch["IE_inputs"]['input_ids'].to(device)
    batch["IE_inputs"]['attention_mask'] = batch["IE_inputs"]['attention_mask'].to(device)
    batch["adj_matrix"] = batch["adj_matrix"].to(device)
    return batch


def run_inference_on_data(mate_model, masc_model, data: list, image_features: dict, 
                          tokenizers: dict, device: str) -> list:
    """Run MABSA inference on data."""
    results = []
    
    for sample in tqdm(data, desc="Running inference"):
        try:
            batch = prepare_single_sample(sample, image_features, tokenizers)
            original_text = batch.pop("original_text")
            original_aspects = batch.pop("original_aspects")
            
            batch = move_batch_to_device(batch, device)
            
            with torch.no_grad():
                # MATE: Extract aspects
                mate_output = mate_model(batch, no_its_and_itm=True)
                
                # Get predicted aspects from MATE output
                n_pred = int(mate_output.n_pred) if hasattr(mate_output, 'n_pred') else 0
                
                # MASC: Classify sentiment (using original aspects as ground truth)
                aspects_with_sentiment = []
                for asp in original_aspects:
                    # For inference, we report the aspect with a default sentiment
                    # In real scenarios, you'd use MASC to predict the sentiment
                    aspects_with_sentiment.append({
                        "term": asp["term"],
                        "position": [asp["from"], asp["to"]],
                        "sentiment": "UNKNOWN",  # Would be predicted by MASC
                        "confidence": 0.0
                    })
            
            result = {
                "text": original_text,
                "aspects_found": n_pred,
                "aspects": aspects_with_sentiment
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            results.append({
                "text": sample.get("review", ""),
                "error": str(e),
                "aspects": []
            })
    
    return results


def inference_from_pkl(mate_model, masc_model, data_path: str, device: str) -> list:
    """Run inference using existing pkl dataset."""
    IE_tokenizer = BertTokenizer.from_pretrained('./Text_encoder/model_best')
    PQ_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = twitter_dataset(
        data_path=data_path,
        IE_tokenizer=IE_tokenizer,
        PQ_former_tokenizer=PQ_tokenizer,
        task='MABSA',
        set_size=1,
        max_seq_len=512,
        num_query_token=32,
        SEP_token_id=2,
        split_token_id=187284
    )
    dataset.update_data()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    results = []
    total_correct = 0
    total_pred = 0
    total_label = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            batch = move_batch_to_device(batch, device)
            
            # MATE
            mate_output = mate_model(batch, no_its_and_itm=True)
            
            if hasattr(mate_output, 'new_batch') and mate_output.new_batch is not None:
                # MASC on extracted aspects
                masc_output = masc_model(mate_output.new_batch, no_its_and_itm=True)
                
                # Handle false predictions
                if hasattr(mate_output, 'false_batch') and mate_output.false_batch is not None:
                    false_output = masc_model(mate_output.false_batch, no_its_and_itm=True)
                    n_correct = masc_output.n_correct - false_output.n_correct
                else:
                    n_correct = masc_output.n_correct
                
                total_correct += n_correct
            
            total_pred += mate_output.n_pred if hasattr(mate_output, 'n_pred') else 0
            total_label += mate_output.n_label if hasattr(mate_output, 'n_label') else 0
            
            results.append({
                "n_pred": int(mate_output.n_pred) if hasattr(mate_output, 'n_pred') else 0,
                "n_label": int(mate_output.n_label) if hasattr(mate_output, 'n_label') else 0,
                "n_correct": int(mate_output.n_correct) if hasattr(mate_output, 'n_correct') else 0
            })
    
    # Calculate metrics
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_label if total_label > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    summary = {
        "total_samples": len(results),
        "total_correct": int(total_correct),
        "total_pred": int(total_pred),
        "total_label": int(total_label),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2)
    }
    
    return {"summary": summary, "samples": results}


def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    mate_model, masc_model = load_models(
        args.mate_model, 
        args.masc_model, 
        device,
        args.gcn_layers
    )
    print("Models loaded!")
    
    # Check input type
    if args.input.endswith('.json'):
        # JSON input - need image cache
        if not args.image_cache or not os.path.exists(args.image_cache):
            print("Error: For JSON input, --image_cache is required")
            print("Run cache_images.py first to create image cache")
            return
        
        print(f"Loading image cache from {args.image_cache}...")
        with open(args.image_cache, 'rb') as f:
            image_features = pickle.load(f)
        
        print(f"Loading input JSON: {args.input}...")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load tokenizers
        tokenizers = {
            'IE': BertTokenizer.from_pretrained('./Text_encoder/model_best'),
            'PQ': BertTokenizer.from_pretrained('bert-base-uncased')
        }
        
        # Run inference
        results = run_inference_on_data(
            mate_model, masc_model, data, image_features, tokenizers, device
        )
        
    else:
        # PKL dataset input
        print(f"Running inference on PKL dataset: {args.input}")
        results = inference_from_pkl(mate_model, masc_model, args.input, device)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary if available
    if isinstance(results, dict) and "summary" in results:
        print("\n=== MABSA Results ===")
        for k, v in results["summary"].items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASCO MABSA Inference")
    parser.add_argument("--mate_model", type=str, required=True,
                        help="Path to MATE model checkpoint")
    parser.add_argument("--masc_model", type=str, required=True,
                        help="Path to MASC model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file (JSON or PKL dataset directory)")
    parser.add_argument("--image_cache", type=str, default="./image_cache/features.pkl",
                        help="Path to image features cache (required for JSON input)")
    parser.add_argument("--output", type=str, default="./mabsa_results.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--gcn_layers", type=int, default=5,
                        help="Number of GCN layers")
    args = parser.parse_args()
    main(args)
