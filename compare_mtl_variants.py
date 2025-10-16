#!/usr/bin/env python3
"""
Compare MTL Variants
다양한 MTL 실험들 간의 비교 (예: Original MTL vs Shared Decoder MTL)
"""
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_test_metrics(log_file):
    """Extract test metrics from training log"""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Find test results section
        if "Test metric" in content or "test_" in content:
            test_section = content[content.rfind("Test metric"):]
            
            # Extract all metrics
            patterns = {
                "test_miou": r"test_miou.*?│\s+([\d.]+)",
                "test_abs_rel": r"test_abs_rel.*?│\s+([\d.]+)",
                "test_delta1": r"test_delta1.*?│\s+([\d.]+)",
                "test_rmse": r"test_rmse.*?│\s+([\d.]+)",
                "test_fps": r"test_fps.*?│\s+([\d.]+)",
                "test_latency": r"test_latency.*?│\s+([\d.]+)",
                "test_loss": r"test_loss.*?│\s+([\d.]+)",
                "test_acc": r"test_acc.*?│\s+([\d.]+)",
            }
            
            # Extract class-wise IoU (validation 기준: val_class_iou_)
            class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
            for class_name in class_names:
                pattern = f"val_class_iou_{class_name}.*?│\s+([\d.]+)"
                match = re.search(pattern, content)  # 전체 content에서 검색
                if match:
                    metrics[f"val_class_iou_{class_name}"] = float(match.group(1))
            
            for key, pattern in patterns.items():
                match = re.search(pattern, test_section)
                if match:
                    metrics[key] = float(match.group(1))
        
        # Extract validation metrics (for comparison)
        val_patterns = {
            "val_miou": r"val_miou.*?│\s+([\d.]+)",
            "val_abs_rel": r"val_abs_rel.*?│\s+([\d.]+)",
            "val_acc": r"val_acc.*?│\s+([\d.]+)",
            "val_latency": r"val_latency.*?│\s+([\d.]+)",
        }
        
        for key, pattern in val_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                # 마지막 validation 결과 사용 (최고 성능)
                metrics[key] = float(matches[-1])
                    
        # Extract training info
        if "best_checkpoint" in content:
            best_match = re.search(r"'best_checkpoint.*?: '(.*?)'", content)
            if best_match:
                checkpoint = best_match.group(1)
                # Extract epoch from checkpoint filename
                epoch_match = re.search(r'epoch[=_](\d+)', checkpoint)
                if epoch_match:
                    metrics["best_epoch"] = int(epoch_match.group(1))
                    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics


def extract_metrics_from_tensorboard(exp_dir):
    """Extract metrics from TensorBoard logs"""
    metrics = {}
    
    try:
        import tensorboard as tb
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find TensorBoard log directory
        log_dirs = list(exp_dir.glob("*_logs"))
        if not log_dirs:
            return metrics
        
        log_dir = log_dirs[0] / "version_0"
        if not log_dir.exists():
            return metrics
        
        # Load TensorBoard data
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Extract scalar metrics
        scalar_tags = ea.Tags()['scalars']
        
        # Extract test metrics
        for tag in scalar_tags:
            if tag.startswith('test_'):
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    # Get the last value
                    last_value = scalar_events[-1].value
                    metrics[tag] = last_value
        
        # Extract validation class IoU
        class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
        for class_name in class_names:
            tag = f"val_class_iou_{class_name}"
            if tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    last_value = scalar_events[-1].value
                    metrics[tag] = last_value
        
        # Extract validation metrics
        for tag in scalar_tags:
            if tag.startswith('val_') and not tag.startswith('val_class_iou_'):
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    last_value = scalar_events[-1].value
                    metrics[tag] = last_value
                    
    except Exception as e:
        print(f"Error extracting TensorBoard metrics: {e}")
    
    return metrics


def find_mtl_experiments(experiments_dir):
    """Find all MTL experiments"""
    if not experiments_dir.exists():
        return []
    
    mtl_exps = []
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        # MTL experiments (including esanet_mtl)
        if (exp_dir.name.startswith("mtl_") or 
            exp_dir.name.startswith("esanet_mtl_")):
            mtl_exps.append(exp_dir)
    
    # Sort by timestamp (latest first)
    mtl_exps.sort(key=lambda x: x.name, reverse=True)
    
    return mtl_exps


def main():
    parser = argparse.ArgumentParser(description="Compare MTL Variants")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Directory containing experiments")
    parser.add_argument("--exp1", type=str, required=True,
                       help="First MTL experiment path (e.g., experiments/mtl_v1_20251013_135919)")
    parser.add_argument("--exp2", type=str, required=True,
                       help="Second MTL experiment path (e.g., experiments/esanet_mtl_resnet34_20251016_123440)")
    parser.add_argument("--name1", type=str, default="Segformer",
                       help="Display name for first experiment")
    parser.add_argument("--name2", type=str, default="ESANet",
                       help="Display name for second experiment")
    parser.add_argument("--output", type=str, default="mtl_variants_comparison.png",
                       help="Output visualization file")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MTL VARIANTS COMPARISON")
    print("=" * 80)
    
    exp1_path = Path(args.exp1)
    exp2_path = Path(args.exp2)
    
    if not exp1_path.exists():
        print(f"❌ First experiment not found: {exp1_path}")
        return
    if not exp2_path.exists():
        print(f"❌ Second experiment not found: {exp2_path}")
        return
    
    print(f"🔍 Comparing:")
    print(f"   Experiment 1: {exp1_path.name} ({args.name1})")
    print(f"   Experiment 2: {exp2_path.name} ({args.name2})")
    
    # Extract metrics from both experiments
    results = {}
    
    # Experiment 1
    if (exp1_path / "training.log").exists():
        results[args.name1] = extract_test_metrics(exp1_path / "training.log")
        print(f"   → Found training.log for {exp1_path.name}")
    else:
        print(f"   → No training.log found for {exp1_path.name}")
        results[args.name1] = {}
    
    # Experiment 2  
    if (exp2_path / "training.log").exists():
        results[args.name2] = extract_test_metrics(exp2_path / "training.log")
        print(f"   → Found training.log for {exp2_path.name}")
    else:
        print(f"   → No training.log found for {exp2_path.name}, trying TensorBoard logs...")
        tb_metrics = extract_metrics_from_tensorboard(exp2_path)
        if tb_metrics:
            results[args.name2] = tb_metrics
            print(f"   → Found TensorBoard metrics for {exp2_path.name}")
        else:
            results[args.name2] = {}
            print(f"   → No metrics found for {exp2_path.name}")
    
    # Check availability
    available = [k for k, v in results.items() if v]
    print(f"\n✅ Available results: {', '.join(available)}")
    
    if len(available) < 2:
        print("\n⚠️  Need both experiments to have results for comparison")
        return
    
    # ========================================================================
    # PERFORMANCE COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Segmentation comparison
    seg_methods = []
    seg_miou = []
    seg_acc = []
    seg_fps = []
    seg_latency = []
    
    for method in [args.name1, args.name2]:
        if method in results:
            # Use validation mIoU if available, otherwise test mIoU
            miou_key = "val_miou" if "val_miou" in results[method] else "test_miou"
            acc_key = "val_acc" if "val_acc" in results[method] else "test_acc"
            
            if miou_key in results[method]:
                # Use validation FPS if available, otherwise test FPS
                fps_key = "val_fps" if "val_fps" in results[method] else "test_fps"
                latency_key = "val_latency" if "val_latency" in results[method] else "test_latency"
                
                seg_methods.append(method)
                seg_miou.append(results[method].get(miou_key, 0))
                seg_acc.append(results[method].get(acc_key, 0))
                seg_fps.append(results[method].get(fps_key, 0))
                seg_latency.append(results[method].get(latency_key, 0))
    
    if len(seg_methods) == 2:
        print(f"\n{'Method':<20} {'mIoU':<12} {'Accuracy':<12} {'FPS':<12} {'Latency(ms)':<12}")
        print("-" * 85)
        
        for i, method in enumerate(seg_methods):
            print(f"{method:<20} {seg_miou[i]:<12.4f} {seg_acc[i]:<12.4f} {seg_fps[i]:<12.1f} {seg_latency[i]:<12.2f}")
        
        improvement = ((seg_miou[1] - seg_miou[0]) / seg_miou[0]) * 100
        print(f"\nSegmentation: {args.name2} vs {args.name1} = {improvement:+.2f}%")
    
    # Depth comparison
    depth_methods = []
    depth_absrel = []
    depth_sqrel = []
    depth_rmse = []
    depth_rmse_log = []
    depth_delta1 = []
    depth_delta2 = []
    depth_delta3 = []
    depth_fps = []
    depth_latency = []
    
    for method in [args.name1, args.name2]:
        if method in results:
            # Use validation AbsRel if available, otherwise test AbsRel
            absrel_key = "val_abs_rel" if "val_abs_rel" in results[method] else "test_abs_rel"
            sqrel_key = "val_sq_rel" if "val_sq_rel" in results[method] else "test_sq_rel"
            
            if absrel_key in results[method]:
                # Use validation metrics if available, otherwise test metrics
                rmse_key = "val_rmse" if "val_rmse" in results[method] else "test_rmse"
                rmse_log_key = "val_rmse_log" if "val_rmse_log" in results[method] else "test_rmse_log"
                delta1_key = "val_delta1" if "val_delta1" in results[method] else "test_delta1"
                delta2_key = "val_delta2" if "val_delta2" in results[method] else "test_delta2"
                delta3_key = "val_delta3" if "val_delta3" in results[method] else "test_delta3"
                fps_key = "val_fps" if "val_fps" in results[method] else "test_fps"
                latency_key = "val_latency" if "val_latency" in results[method] else "test_latency"
                
                depth_methods.append(method)
                depth_absrel.append(results[method].get(absrel_key, 0.0))
                depth_sqrel.append(results[method].get(sqrel_key, 0.0))
                depth_rmse.append(results[method].get(rmse_key, 0.0))
                depth_rmse_log.append(results[method].get(rmse_log_key, 0.0))
                depth_delta1.append(results[method].get(delta1_key, 0.0))
                depth_delta2.append(results[method].get(delta2_key, 0.0))
                depth_delta3.append(results[method].get(delta3_key, 0.0))
                depth_fps.append(results[method].get(fps_key, 0.0))
                depth_latency.append(results[method].get(latency_key, 0.0))
    
    if len(depth_methods) == 2:
        print(f"\n{'Method':<20} {'AbsRel':<10} {'SqRel':<10} {'RMSE':<10} {'RMSElog':<10} {'δ<1.25':<10} {'δ<1.25²':<10} {'δ<1.25³':<10} {'FPS':<10} {'Latency(ms)':<12}")
        print("-" * 135)
        
        for i, method in enumerate(depth_methods):
            print(f"{method:<20} {depth_absrel[i]:<10.4f} {depth_sqrel[i]:<10.4f} {depth_rmse[i]:<10.4f} {depth_rmse_log[i]:<10.4f} {depth_delta1[i]:<10.4f} {depth_delta2[i]:<10.4f} {depth_delta3[i]:<10.4f} {depth_fps[i]:<10.1f} {depth_latency[i]:<12.2f}")
        
        improvement = ((depth_absrel[0] - depth_absrel[1]) / depth_absrel[0]) * 100
        print(f"\nDepth: {args.name2} vs {args.name1} = {improvement:+.2f}% (negative is better)")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    if seg_methods and depth_methods:
        # Create main comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Segmentation Performance
        ax = axes[0, 0]
        x = np.arange(len(seg_methods))
        bars = ax.bar(x, seg_miou, color=['#3498db', '#e74c3c'], width=0.6)
        ax.set_ylabel('mIoU (higher is better)', fontsize=12, fontweight='bold')
        ax.set_title('Segmentation Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seg_methods, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, seg_miou)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Depth Performance
        ax = axes[0, 1]
        x = np.arange(len(depth_methods))
        bars = ax.bar(x, depth_absrel, color=['#2ecc71', '#e74c3c'], width=0.6)
        ax.set_ylabel('AbsRel (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('Depth Estimation Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(depth_methods, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, depth_absrel)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Class-wise IoU Comparison
        if args.name1 in results and args.name2 in results:
            ax = axes[1, 0]
            class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
            exp1_class_iou = []
            exp2_class_iou = []
            
            for class_name in class_names:
                exp1_key = f"val_class_iou_{class_name}"
                exp2_key = f"val_class_iou_{class_name}"
                exp1_val = results[args.name1].get(exp1_key, 0)
                exp2_val = results[args.name2].get(exp2_key, 0)
                exp1_class_iou.append(exp1_val)
                exp2_class_iou.append(exp2_val)
            
            x = np.arange(len(class_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, exp1_class_iou, width, label=args.name1, color='#3498db')
            bars2 = ax.bar(x + width/2, exp2_class_iou, width, label=args.name2, color='#e74c3c')
            
            ax.set_ylabel('IoU (higher is better)', fontsize=12, fontweight='bold')
            ax.set_title('Class-wise IoU Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Set y-axis limits to show small values
            max_val = max(max(exp1_class_iou), max(exp2_class_iou))
            if max_val > 0:
                ax.set_ylim(0, max_val * 1.2)
            else:
                ax.set_ylim(0, 0.1)  # Default range if all values are 0
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Speed Comparison
        ax = axes[1, 1]
        
        if args.name1 in results and args.name2 in results:
            methods = [args.name1, args.name2]
            fps_values = [results[args.name1].get("test_fps", 0), results[args.name2].get("test_fps", 0)]
            colors = ['#3498db', '#e74c3c']
            
            x = np.arange(len(methods))
            bars = ax.bar(x, fps_values, color=colors, width=0.6)
            ax.set_ylabel('FPS (higher is better)', fontsize=12, fontweight='bold')
            ax.set_title('Speed Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels and speedup
            for i, (bar, val) in enumerate(zip(bars, fps_values)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            if fps_values[0] > 0 and fps_values[1] > 0:
                speedup = fps_values[1] / fps_values[0]
                
                if speedup > 1:
                    text = f'{args.name2} is {speedup:.2f}x faster'
                    color = "lightgreen"
                else:
                    slowdown = 1 / speedup
                    text = f'{args.name1} is {slowdown:.2f}x faster'
                    color = "lightcoral"
                
                ax.text(0.5, 0.95, text, 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=11, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\n📊 Main comparison saved: {args.output}")
        
        # Create additional depth metrics comparison figure
        if len(depth_methods) == 2:
            fig_depth, axes_depth = plt.subplots(2, 3, figsize=(18, 12))
            
            # AbsRel comparison
            ax = axes_depth[0, 0]
            x = np.arange(len(depth_methods))
            bars = ax.bar(x, depth_absrel, color=['#2ecc71', '#e74c3c'], width=0.6)
            ax.set_ylabel('AbsRel (lower is better)', fontsize=12, fontweight='bold')
            ax.set_title('Absolute Relative Error', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, depth_absrel)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # SqRel comparison
            ax = axes_depth[0, 1]
            bars = ax.bar(x, depth_sqrel, color=['#2ecc71', '#e74c3c'], width=0.6)
            ax.set_ylabel('SqRel (lower is better)', fontsize=12, fontweight='bold')
            ax.set_title('Squared Relative Error', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, depth_sqrel)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # RMSE comparison
            ax = axes_depth[0, 2]
            bars = ax.bar(x, depth_rmse, color=['#2ecc71', '#e74c3c'], width=0.6)
            ax.set_ylabel('RMSE (lower is better)', fontsize=12, fontweight='bold')
            ax.set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, depth_rmse)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # RMSElog comparison
            ax = axes_depth[1, 0]
            bars = ax.bar(x, depth_rmse_log, color=['#2ecc71', '#e74c3c'], width=0.6)
            ax.set_ylabel('RMSElog (lower is better)', fontsize=12, fontweight='bold')
            ax.set_title('RMSE in Log Space', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, depth_rmse_log)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Delta metrics comparison (δ<1.25, δ<1.25², δ<1.25³)
            ax = axes_depth[1, 1]
            delta_metrics = [depth_delta1, depth_delta2, depth_delta3]
            delta_names = ['δ<1.25', 'δ<1.25²', 'δ<1.25³']
            delta_colors = ['#3498db', '#9b59b6', '#e67e22']
            
            x_delta = np.arange(len(delta_names))
            width = 0.35
            
            for i, (method, metrics) in enumerate(zip(depth_methods, zip(*delta_metrics))):
                ax.bar(x_delta + i*width, metrics, width, label=method, 
                      color=['#2ecc71', '#e74c3c'][i], alpha=0.8)
            
            ax.set_ylabel('Accuracy (higher is better)', fontsize=12, fontweight='bold')
            ax.set_title('Delta Accuracy Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x_delta + width/2)
            ax.set_xticklabels(delta_names, fontsize=11)
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.0)
            
            # Add value labels for delta metrics
            for i, (method, metrics) in enumerate(zip(depth_methods, zip(*delta_metrics))):
                for j, (x_pos, val) in enumerate(zip(x_delta + i*width, metrics)):
                    ax.text(x_pos, val + 0.01, f'{val:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
            
            # Speed comparison
            ax = axes_depth[1, 2]
            bars = ax.bar(x, depth_fps, color=['#f39c12', '#e74c3c'], width=0.6)
            ax.set_ylabel('FPS (higher is better)', fontsize=12, fontweight='bold')
            ax.set_title('Inference Speed', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_methods, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for i, (bar, val) in enumerate(zip(bars, depth_fps)):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            depth_output = args.output.replace('.png', '_depth_metrics.png')
            plt.savefig(depth_output, dpi=150, bbox_inches='tight')
            print(f"📊 Depth metrics comparison saved: {depth_output}")
            plt.close()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
