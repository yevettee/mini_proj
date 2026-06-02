"""
Experiment Result Analysis and Visualization
- Generate comparison charts by model/condition
- Generate final report
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Allow direct execution: insert scripts dir for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PROJECT_ROOT, EXPERIMENT_GROUPS


# Color palette
MODEL_COLORS = {
    'v8': '#3498db',   # 파랑
    'v11': '#2ecc71',  # 녹색
    'v26': '#e74c3c',  # 빨강
}

def get_model_color(exp_name: str) -> str:
    """Get model color from experiment name"""
    if 'v26' in exp_name.lower():
        return MODEL_COLORS['v26']
    elif 'v11' in exp_name.lower():
        return MODEL_COLORS['v11']
    else:
        return MODEL_COLORS['v8']


def get_model_version(exp_name: str) -> str:
    """Extract model version from experiment name"""
    if 'v26' in exp_name.lower():
        return 'YOLO26'
    elif 'v11' in exp_name.lower():
        return 'YOLO11'
    elif 'v8' in exp_name.lower():
        return 'YOLOv8'
    return 'Unknown'


def load_results(results_csv: str) -> pd.DataFrame:
    """Load results CSV"""
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    df['model_version'] = df['experiment_name'].apply(get_model_version)
    return df


def plot_baseline_comparison(df: pd.DataFrame, output_dir: str):
    """Baseline comparison chart - main metrics by model"""
    baseline_df = df[df['experiment_name'].str.contains('baseline')]
    
    if baseline_df.empty:
        print("Warning: No baseline experiment results")
        return
    
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
    metric_labels = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        colors = [get_model_color(name) for name in baseline_df['experiment_name']]
        bars = ax.bar(baseline_df['model_version'], baseline_df[metric], color=colors)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        
        # 값 표시
        for bar, val in zip(bars, baseline_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Baseline Model Comparison (imgsz=640, no augment)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_imgsz_comparison(df: pd.DataFrame, output_dir: str):
    """Performance comparison by image size"""
    imgsz_df = df[df['imgsz'].isin([320, 480, 640]) & (df['preprocess'] == 'none')]
    
    if imgsz_df.empty:
        print("Warning: No image size comparison results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # mAP50 비교
    for model in ['YOLOv8', 'YOLO11', 'YOLO26']:
        model_df = imgsz_df[imgsz_df['model_version'] == model].sort_values('imgsz')
        if not model_df.empty:
            color = MODEL_COLORS.get(model.lower().replace('yolo', '').replace('v', ''), '#333')
            axes[0].plot(model_df['imgsz'], model_df['mAP50'], 'o-', 
                        label=model, color=color, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Image Size', fontsize=11)
    axes[0].set_ylabel('mAP@50', fontsize=11)
    axes[0].set_title('mAP@50 vs Image Size', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([320, 480, 640])
    
    # mAP50-95 비교
    for model in ['YOLOv8', 'YOLO11', 'YOLO26']:
        model_df = imgsz_df[imgsz_df['model_version'] == model].sort_values('imgsz')
        if not model_df.empty:
            color = MODEL_COLORS.get(model.lower().replace('yolo', '').replace('v', ''), '#333')
            axes[1].plot(model_df['imgsz'], model_df['mAP50_95'], 'o-',
                        label=model, color=color, linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Image Size', fontsize=11)
    axes[1].set_ylabel('mAP@50-95', fontsize=11)
    axes[1].set_title('mAP@50-95 vs Image Size', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([320, 480, 640])
    
    plt.suptitle('Image Size Impact on Model Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'imgsz_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_degradation_comparison(df: pd.DataFrame, output_dir: str):
    """Performance comparison under quality degradation (blur/noise)"""
    # Get baseline performance
    baseline_df = df[df['experiment_name'].str.contains('baseline')].copy()
    baseline_map = dict(zip(baseline_df['model_version'], baseline_df['mAP50']))
    
    # Filter blur experiments
    blur_df = df[df['preprocess'].str.contains('blur', na=False)].copy()
    
    if blur_df.empty:
        print("Warning: No blur experiment results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Blur 강도별 성능 저하율
    for model in ['YOLOv8', 'YOLO11', 'YOLO26']:
        model_blur = blur_df[blur_df['model_version'] == model]
        if model_blur.empty:
            continue
        
        baseline = baseline_map.get(model, 1.0)
        
        # Separate light/heavy blur
        light_blur = model_blur[model_blur['experiment_name'].str.contains('light')]
        heavy_blur = model_blur[model_blur['experiment_name'].str.contains('heavy')]
        
        points = [(0, baseline)]  # 베이스라인
        if not light_blur.empty:
            points.append((1.5, light_blur['mAP50'].values[0]))
        if not heavy_blur.empty:
            points.append((2.5, heavy_blur['mAP50'].values[0]))
        
        points.sort(key=lambda x: x[0])
        x_vals, y_vals = zip(*points)
        
        color = MODEL_COLORS.get(model.lower().replace('yolo', '').replace('v', ''), '#333')
        axes[0].plot(x_vals, y_vals, 'o-', label=model, color=color, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Blur Sigma', fontsize=11)
    axes[0].set_ylabel('mAP@50', fontsize=11)
    axes[0].set_title('mAP@50 vs Blur Intensity', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([0, 1.5, 2.5])
    axes[0].set_xticklabels(['Baseline', 'Light (σ=1.5)', 'Heavy (σ=2.5)'])
    
    # Performance degradation comparison (bar chart)
    degradation_data = []
    for model in ['YOLOv8', 'YOLO11', 'YOLO26']:
        baseline = baseline_map.get(model, 1.0)
        heavy_blur = blur_df[(blur_df['model_version'] == model) & 
                             (blur_df['experiment_name'].str.contains('heavy'))]
        if not heavy_blur.empty:
            blur_map = heavy_blur['mAP50'].values[0]
            degradation = ((baseline - blur_map) / baseline) * 100
            degradation_data.append({'model': model, 'degradation': degradation})
    
    if degradation_data:
        deg_df = pd.DataFrame(degradation_data)
        colors = [MODEL_COLORS.get(m.lower().replace('yolo', '').replace('v', ''), '#333') 
                 for m in deg_df['model']]
        bars = axes[1].bar(deg_df['model'], deg_df['degradation'], color=colors)
        
        for bar, val in zip(bars, deg_df['degradation']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        axes[1].set_ylabel('Performance Drop (%)', fontsize=11)
        axes[1].set_title('Performance Degradation under Heavy Blur', fontsize=12, fontweight='bold')
        axes[1].set_ylim(0, max(deg_df['degradation']) * 1.3)
    
    plt.suptitle('Model Robustness to Image Quality Degradation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'degradation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_augmentation_effect(df: pd.DataFrame, output_dir: str):
    """Augmentation effect comparison"""
    baseline_df = df[df['experiment_name'].str.contains('baseline')]
    augment_df = df[df['experiment_name'].str.contains('augment')]
    
    if augment_df.empty:
        print("Warning: No augmentation experiment results")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(3)  # 3개 모델
    width = 0.35
    
    models = ['YOLOv8', 'YOLO11', 'YOLO26']
    baseline_vals = []
    augment_vals = []
    
    for model in models:
        bl = baseline_df[baseline_df['model_version'] == model]['mAP50']
        ag = augment_df[augment_df['model_version'] == model]['mAP50']
        baseline_vals.append(bl.values[0] if len(bl) > 0 else 0)
        augment_vals.append(ag.values[0] if len(ag) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='No Augmentation', color='#95a5a6')
    bars2 = ax.bar(x + width/2, augment_vals, width, label='Mosaic Augmentation', color='#27ae60')
    
    ax.set_ylabel('mAP@50', fontsize=11)
    ax.set_title('Effect of Mosaic Augmentation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Show values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'augmentation_effect.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_size_comparison(df: pd.DataFrame, output_dir: str):
    """Model size comparison (nano vs small)"""
    nano_df = df[df['model'].str.contains('n.pt', na=False)]
    small_df = df[df['model'].str.contains('s.pt', na=False)]
    
    if small_df.empty:
        print("Warning: No small model experiment results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['YOLOv8', 'YOLO11', 'YOLO26']
    x = np.arange(len(models))
    width = 0.35
    
    # mAP50-95 비교
    nano_vals = []
    small_vals = []
    for model in models:
        n = nano_df[(nano_df['model_version'] == model) & 
                   (nano_df['experiment_name'].str.contains('baseline'))]['mAP50_95']
        s = small_df[small_df['model_version'] == model]['mAP50_95']
        nano_vals.append(n.values[0] if len(n) > 0 else 0)
        small_vals.append(s.values[0] if len(s) > 0 else 0)
    
    bars1 = axes[0].bar(x - width/2, nano_vals, width, label='Nano', color='#3498db')
    bars2 = axes[0].bar(x + width/2, small_vals, width, label='Small', color='#9b59b6')
    
    axes[0].set_ylabel('mAP@50-95', fontsize=11)
    axes[0].set_title('Accuracy: Nano vs Small', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Inference speed comparison (if available)
    if 'inference_ms' in df.columns:
        nano_speed = []
        small_speed = []
        for model in models:
            n = nano_df[(nano_df['model_version'] == model) & 
                       (nano_df['experiment_name'].str.contains('baseline'))]['inference_ms']
            s = small_df[small_df['model_version'] == model]['inference_ms']
            nano_speed.append(n.values[0] if len(n) > 0 and not pd.isna(n.values[0]) else 0)
            small_speed.append(s.values[0] if len(s) > 0 and not pd.isna(s.values[0]) else 0)
        
        bars1 = axes[1].bar(x - width/2, nano_speed, width, label='Nano', color='#3498db')
        bars2 = axes[1].bar(x + width/2, small_speed, width, label='Small', color='#9b59b6')
        
        axes[1].set_ylabel('Inference Time (ms)', fontsize=11)
        axes[1].set_title('Speed: Nano vs Small', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1].text(0.5, 0.5, 'Inference time data not available',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Speed: Nano vs Small', fontsize=12, fontweight='bold')
    
    plt.suptitle('Model Size Trade-off Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'model_size_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comprehensive_heatmap(df: pd.DataFrame, output_dir: str):
    """Comprehensive experiment results heatmap"""
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
    
    # Filter successful experiments only
    success_df = df[df['status'] == 'success'].copy()
    
    if success_df.empty:
        print("Warning: No successful experiments")
        return
    
    # Create pivot table
    pivot_data = success_df.set_index('experiment_name')[metrics]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.4)))
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Score'})
    
    ax.set_title('Comprehensive Experiment Results', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Experiment', fontsize=11)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'comprehensive_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generate summary table image"""
    success_df = df[df['status'] == 'success'].copy()
    
    if success_df.empty:
        print("Warning: No successful experiments")
        return
    
    # Select main columns
    cols = ['experiment_name', 'model_version', 'imgsz', 'preprocess', 
            'mAP50', 'mAP50_95', 'f1_score', 'train_time_min']
    
    # Add train_time_min if not exists
    if 'train_time_min' not in success_df.columns and 'train_time_sec' in success_df.columns:
        success_df['train_time_min'] = success_df['train_time_sec'] / 60
    elif 'train_time_min' not in success_df.columns:
        success_df['train_time_min'] = 0
    
    summary_df = success_df[cols].copy()
    summary_df.columns = ['Experiment', 'Model', 'ImgSz', 'Preprocess',
                         'mAP50', 'mAP50-95', 'F1', 'Time(min)']
    
    # Number formatting
    for col in ['mAP50', 'mAP50-95', 'F1']:
        summary_df[col] = summary_df[col].apply(lambda x: f'{x:.4f}')
    summary_df['Time(min)'] = summary_df['Time(min)'].apply(lambda x: f'{x:.1f}')
    
    fig, ax = plt.subplots(figsize=(16, max(4, len(summary_df) * 0.4)))
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.09, 0.07, 0.12, 0.09, 0.09, 0.08, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Header style
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#EAF2FB')
            else:
                table[(i, j)].set_facecolor('#FDFEFE')
    
    ax.set_title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_best_model_report(df: pd.DataFrame, output_dir: str):
    """Generate best model report"""
    success_df = df[df['status'] == 'success'].copy()
    
    if success_df.empty:
        print("Warning: No successful experiments")
        return
    
    report_lines = [
        "=" * 60,
        " YOLO Model Comparison - Best Model Report",
        f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    
    # Best performance by metric
    metrics = {
        'mAP50': 'mAP@50',
        'mAP50_95': 'mAP@50-95',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    }
    
    report_lines.append("[ Best Performance by Metric ]")
    report_lines.append("-" * 40)
    
    for col, name in metrics.items():
        best_idx = success_df[col].idxmax()
        best_row = success_df.loc[best_idx]
        report_lines.append(f"{name}: {best_row[col]:.4f}")
        report_lines.append(f"  -> {best_row['experiment_name']} ({best_row['model_version']})")
        report_lines.append("")
    
    # Overall recommendation
    report_lines.append("-" * 40)
    report_lines.append("[ Overall Recommendation ]")
    report_lines.append("")
    
    # Best mAP50-95 in baseline
    baseline_df = success_df[success_df['experiment_name'].str.contains('baseline')]
    if not baseline_df.empty:
        best_baseline = baseline_df.loc[baseline_df['mAP50_95'].idxmax()]
        report_lines.append(f"Best Baseline Model: {best_baseline['model_version']}")
        report_lines.append(f"  mAP@50-95: {best_baseline['mAP50_95']:.4f}")
        report_lines.append("")
    
    # Average performance by model
    report_lines.append("[ Average Performance by Model Version ]")
    model_avg = success_df.groupby('model_version')[['mAP50', 'mAP50_95', 'f1_score']].mean()
    for model in model_avg.index:
        row = model_avg.loc[model]
        report_lines.append(f"{model}:")
        report_lines.append(f"  Avg mAP50: {row['mAP50']:.4f} | Avg mAP50-95: {row['mAP50_95']:.4f} | Avg F1: {row['f1_score']:.4f}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    # Save
    report_text = "\n".join(report_lines)
    
    report_path = os.path.join(output_dir, 'best_model_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"Saved: {report_path}")
    print("\n" + report_text)


def plot_training_time_comparison(df: pd.DataFrame, output_dir: str):
    """Training time comparison chart"""
    success_df = df[df['status'] == 'success'].copy()
    
    if success_df.empty or 'train_time_sec' not in success_df.columns:
        print("Warning: No training time data available")
        return
    
    # Convert to minutes
    success_df['train_time_min'] = success_df['train_time_sec'] / 60
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Baseline training time comparison
    baseline_df = success_df[success_df['experiment_name'].str.contains('baseline')]
    if not baseline_df.empty:
        colors = [get_model_color(name) for name in baseline_df['experiment_name']]
        bars = axes[0].bar(baseline_df['model_version'], baseline_df['train_time_min'], color=colors)
        axes[0].set_ylabel('Training Time (min)', fontsize=11)
        axes[0].set_title('Baseline Training Time', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, baseline_df['train_time_min']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Training time by image size
    imgsz_df = success_df[success_df['preprocess'] == 'none']
    if not imgsz_df.empty:
        for model in ['YOLOv8', 'YOLO11', 'YOLO26']:
            model_df = imgsz_df[imgsz_df['model_version'] == model].sort_values('imgsz')
            if not model_df.empty and len(model_df) > 1:
                color = MODEL_COLORS.get(model.lower().replace('yolo', '').replace('v', ''), '#333')
                axes[1].plot(model_df['imgsz'], model_df['train_time_min'], 'o-',
                            label=model, color=color, linewidth=2, markersize=8)
        
        axes[1].set_xlabel('Image Size', fontsize=11)
        axes[1].set_ylabel('Training Time (min)', fontsize=11)
        axes[1].set_title('Training Time vs Image Size', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        if len(imgsz_df['imgsz'].unique()) > 1:
            axes[1].set_xticks(sorted(imgsz_df['imgsz'].unique()))
    
    plt.suptitle('Training Time Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_time_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_all(results_csv: str, output_dir: str):
    """Run all analysis"""
    print(f"\n{'='*60}")
    print(f" YOLO Experiment Result Analysis")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = load_results(results_csv)
        print(f"Loaded experiments: {len(df)}")
        print(f"Success: {len(df[df['status']=='success'])} | Failed: {len(df[df['status']=='failed'])}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("\nGenerating charts...")
    
    # Generate all charts
    plot_baseline_comparison(df, output_dir)
    plot_imgsz_comparison(df, output_dir)
    plot_degradation_comparison(df, output_dir)
    plot_augmentation_effect(df, output_dir)
    plot_model_size_comparison(df, output_dir)
    plot_training_time_comparison(df, output_dir)
    plot_comprehensive_heatmap(df, output_dir)
    generate_summary_table(df, output_dir)
    generate_best_model_report(df, output_dir)
    
    print(f"\nAnalysis complete! Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Experiment Result Analysis")
    parser.add_argument("--results", type=str, default=os.path.join(PROJECT_ROOT, "summary.csv"),
                       help="Results CSV path")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "comparison_charts"),
                       help="Chart output path")
    
    args = parser.parse_args()
    
    analyze_all(args.results, args.output)


if __name__ == "__main__":
    main()
