"""
YOLO Model Comparison Experiment Pipeline Runner
- Run configured experiments sequentially
- Save results to CSV
- Resume after interruption
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Allow direct execution (python runner.py) or as module.
# Insert scripts/ dir so sibling modules (config, preprocess) are importable by name.
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SEED, EPOCHS, PATIENCE, BATCH_SIZE, DATA_YAML, PROJECT_ROOT, RUNS_ROOT,
    EXPERIMENTS, EXPERIMENT_GROUPS, get_experiments
)
from preprocess import get_data_yaml_for_experiment


def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] Reproducibility seed set (SEED={seed})")


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = 0
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[Device] Running on CPU")
    return device


def load_completed_experiments(results_csv: str) -> set:
    """Load list of completed experiments"""
    if not os.path.exists(results_csv):
        return set()
    
    try:
        df = pd.read_csv(results_csv)
        return set(df['experiment_name'].tolist())
    except Exception:
        return set()


def save_result(results_csv: str, result: dict):
    """Append single experiment result to CSV"""
    df_new = pd.DataFrame([result])
    
    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(results_csv, index=False)
    print(f"[Save] Result saved: {results_csv}")


def run_single_experiment(
    exp_config: dict,
    base_data_yaml: str,
    device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict:
    """Run single experiment
    
    Returns:
        Experiment result dictionary
    """
    exp_name = exp_config["name"]
    model_name = exp_config["model"]
    imgsz = exp_config["imgsz"]
    augment = exp_config.get("augment", False)
    description = exp_config.get("description", "")
    
    print("\n" + "=" * 60)
    print(f" Experiment: {exp_name}")
    print(f" Model: {model_name} | ImageSize: {imgsz} | Augment: {augment}")
    print(f" Description: {description}")
    print("=" * 60)
    
    # Prepare dataset if preprocessing required
    data_yaml = get_data_yaml_for_experiment(exp_config, base_data_yaml)
    print(f"[Data] Dataset: {data_yaml}")
    
    # Reset seed for each experiment
    set_seed(seed)
    
    # Training artifacts (weights, plots, etc.) go under models/runs/model_comparison/
    # (keeps heavy files separate from summary CSVs/charts in results/)
    train_project = os.path.join(RUNS_ROOT, f"seed_{seed}")
    save_name = exp_name
    
    # Load model and train
    start_time = datetime.now()
    
    try:
        model = YOLO(model_name)
        
        # Training
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            mosaic=1.0 if augment else 0.0,
            project=train_project,
            name=save_name,
            device=device,
            plots=True,
            patience=patience,
            seed=seed,
            verbose=True
        )
        
        # Evaluate on test dataset
        print(f"\n[Eval] Evaluating on test dataset...")
        metrics = model.val(data=data_yaml, split='test', device=device, save_json=True)
        
        # Calculate training time
        train_end_time = datetime.now()
        train_time_sec = (train_end_time - start_time).total_seconds()
        
        # Inference speed measurement
        inference_time = metrics.speed.get('inference', 0)
        
        # Collect results
        p = metrics.results_dict['metrics/precision(B)']
        r = metrics.results_dict['metrics/recall(B)']
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        
        result = {
            "experiment_name": exp_name,
            "model": model_name,
            "imgsz": imgsz,
            "preprocess": exp_config.get("preprocess", "none"),
            "preprocess_params": json.dumps(exp_config.get("preprocess_params", {})),
            "augment": augment,
            "epochs": epochs,
            "seed": seed,
            "mAP50": metrics.results_dict['metrics/mAP50(B)'],
            "mAP50_95": metrics.results_dict['metrics/mAP50-95(B)'],
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "inference_ms": inference_time,
            "train_time_sec": train_time_sec,
            "train_time_min": train_time_sec / 60,
            "status": "success",
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add per-class AP if available
        if hasattr(metrics, 'ap_class_index') and metrics.ap_class_index is not None:
            for i, cls_idx in enumerate(metrics.ap_class_index):
                cls_name = metrics.names.get(cls_idx, f"class_{cls_idx}")
                result[f"AP50_{cls_name}"] = float(metrics.box.ap50[i])
        
        print(f"\n[Result] mAP50: {result['mAP50']:.4f} | mAP50-95: {result['mAP50_95']:.4f}")
        print(f"[Result] Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        
    except Exception as e:
        print(f"[Error] Experiment failed: {e}")
        train_time_sec = (datetime.now() - start_time).total_seconds()
        result = {
            "experiment_name": exp_name,
            "model": model_name,
            "imgsz": imgsz,
            "preprocess": exp_config.get("preprocess", "none"),
            "augment": augment,
            "seed": seed,
            "train_time_sec": train_time_sec,
            "train_time_min": train_time_sec / 60,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    return result


def run_pipeline(
    experiments: list = None,
    group: str = None,
    only: str = None,
    skip_completed: bool = True,
    base_data_yaml: str = DATA_YAML,
    seed: int = SEED,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    project_root: str = PROJECT_ROOT
):
    """Run experiment pipeline
    
    Args:
        experiments: List of experiments to run (None for all)
        group: Experiment group name
        only: Run only specific experiment by name
        skip_completed: Skip already completed experiments
        base_data_yaml: Base dataset yaml path
        seed: Seed value
        epochs: Number of epochs
        patience: Early stopping patience
        batch_size: Batch size
        project_root: Results (summary.csv, charts) root. Training runs use RUNS_ROOT.
    """
    # Determine experiment list
    if only:
        exp_list = get_experiments(names=[only])
    elif group:
        exp_list = get_experiments(group=group)
    elif experiments:
        exp_list = experiments
    else:
        exp_list = EXPERIMENTS
    
    if not exp_list:
        print("[Error] No experiments to run.")
        return
    
    print(f"\n{'='*60}")
    print(f" YOLO Model Comparison Experiment Pipeline")
    print(f" Total {len(exp_list)} experiments scheduled")
    print(f" Seed: {seed} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"{'='*60}")
    
    # Results CSV path
    os.makedirs(project_root, exist_ok=True)
    results_csv = os.path.join(project_root, "summary.csv")
    
    # Check already completed experiments
    completed = load_completed_experiments(results_csv) if skip_completed else set()
    if completed:
        print(f"[Skip] Already completed: {len(completed)} experiments")
    
    # Device setup
    device = get_device()
    
    # Run experiments
    for i, exp_config in enumerate(exp_list, 1):
        exp_name = exp_config["name"]
        
        if exp_name in completed:
            print(f"\n[{i}/{len(exp_list)}] {exp_name} - Already done, skipping")
            continue
        
        print(f"\n[{i}/{len(exp_list)}] Running {exp_name}...")
        
        result = run_single_experiment(
            exp_config=exp_config,
            base_data_yaml=base_data_yaml,
            device=device,
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
        )
        
        # Save result
        save_result(results_csv, result)
    
    print(f"\n{'='*60}")
    print(f" Pipeline Complete!")
    print(f" Results: {results_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Model Comparison Experiment Pipeline")
    parser.add_argument("--group", type=str, help="Experiment group (baseline, imgsz_comparison, blur_test, ...)")
    parser.add_argument("--only", type=str, help="Run specific experiment only")
    parser.add_argument("--no-skip", action="store_true", help="Re-run completed experiments")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Seed value (default: {SEED})")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--list-groups", action="store_true", help="List experiment groups")
    
    args = parser.parse_args()
    
    if args.list:
        print("All experiments:")
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"  {i:2d}. {exp['name']}: {exp.get('description', '')}")
        return
    
    if args.list_groups:
        print("Experiment groups:")
        for group, names in EXPERIMENT_GROUPS.items():
            print(f"  - {group}: {names}")
        return
    
    run_pipeline(
        group=args.group,
        only=args.only,
        skip_completed=not args.no_skip,
        epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
