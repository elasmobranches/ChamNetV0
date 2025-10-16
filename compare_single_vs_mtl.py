#!/usr/bin/env python3
"""
Compare Single-Task vs Multi-Task Learning
Seg-Only, Depth-Only vs MTL 비교
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


def find_experiments_by_type(experiments_dir):
    """Find experiments by type"""
    if not experiments_dir.exists():
        return {}, {}, {}
    
    seg_exps = []
    depth_exps = []
    mtl_exps = []
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        if exp_dir.name.startswith("seg_"):
            seg_exps.append(exp_dir)
        elif exp_dir.name.startswith("depth_"):
            depth_exps.append(exp_dir)
        elif exp_dir.name.startswith("mtl_"):
            mtl_exps.append(exp_dir)
    
    # Sort by timestamp (latest first)
    seg_exps.sort(key=lambda x: x.name, reverse=True)
    depth_exps.sort(key=lambda x: x.name, reverse=True)
    mtl_exps.sort(key=lambda x: x.name, reverse=True)
    
    return seg_exps, depth_exps, mtl_exps


def benchmark_sequential_models(seg_checkpoint, depth_checkpoint, test_loader, device):
    """순차적으로 두 모델을 실행하고 FPS 측정 (웜업 포함)"""
    try:
        import torch
        import time
        import numpy as np
        from train_seg_only import LightningSegSegformer
        from train_depth_only import LightningDepthSegformer
        
        print(f"\n🔄 Benchmarking Sequential Models...")
        
        # Load segmentation model
        seg_model = LightningSegSegformer.load_from_checkpoint(
            str(seg_checkpoint), 
            strict=False
        ).to(device)
        seg_model.eval()
        
        # Load depth model  
        depth_model = LightningDepthSegformer.load_from_checkpoint(
            str(depth_checkpoint),
            strict=False
        ).to(device)
        depth_model.eval()
        
        # Warm-up phase (2 batches)
        print(f"   🔥 Warming up (2 batches)...")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_loader):
                if batch_idx >= 2:
                    break
                images = images.to(device)
                seg_logits = seg_model(images)
                depth_pred = depth_model(images)
        
        # Benchmark phase (steady-state measurement)
        print(f"   📊 Measuring steady-state performance...")
        times = []
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_loader):
                if batch_idx >= 10:  # Measure up to 8 batches after warm-up
                    break
                    
                images = images.to(device)
                
                # Sequential execution timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                
                # Run segmentation
                seg_logits = seg_model(images)
                
                # Run depth estimation
                depth_pred = depth_model(images)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                dt = time.perf_counter() - t0
                
                # Only measure after warm-up (skip first 2 batches)
                if batch_idx >= 2:
                    times.append(dt)
        
        # Calculate average FPS (steady-state)
        if times:
            avg_time = np.mean(times)
            avg_fps = test_loader.batch_size / avg_time
            print(f"   Sequential FPS (steady-state): {avg_fps:.1f}")
            return avg_fps
        else:
            print(f"   No valid measurements after warm-up")
            return 0
        
    except Exception as e:
        print(f"Error benchmarking sequential models: {e}")
        return 0


def test_seg_checkpoint(checkpoint_path):
    """Test Segmentation checkpoint and return metrics with accurate FPS measurement"""
    try:
        import torch
        import time
        import numpy as np
        from train_seg_only import LightningSegSegformer, build_seg_datasets
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        
        print(f"\n🔍 Testing Seg checkpoint: {checkpoint_path.name}")
        
        # Load model
        model = LightningSegSegformer.load_from_checkpoint(
            str(checkpoint_path), 
            strict=False
        )
        
        # Load test dataset
        dataset_root = Path("../dataset")
        _, _, test_ds = build_seg_datasets(dataset_root, "mit_b2", (512, 512))
        
        test_loader = DataLoader(
            test_ds, batch_size=4, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        # Test with PyTorch Lightning (for other metrics)
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        results = trainer.test(model, test_loader, verbose=False)
        
        if results and len(results) > 0:
            # Override FPS with accurate measurement
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Measure FPS accurately
            times = []
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(test_loader):
                    if batch_idx >= 6:  # Measure 6 batches
                        break
                    images = images.to(device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    logits = model(images)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    
                    if batch_idx >= 2:  # Skip warm-up
                        times.append(dt)
            
            if times:
                avg_time = np.mean(times)
                accurate_fps = test_loader.batch_size / avg_time
                results[0]["test_fps"] = accurate_fps
                print(f"   → Accurate FPS: {accurate_fps:.1f}")
            
            # Add validation metrics from log file if available
            log_file = checkpoint_path.parent / "training.log"
            if log_file.exists():
                log_metrics = extract_test_metrics(log_file)
                # Add validation metrics to results
                for key, value in log_metrics.items():
                    if key.startswith("val_"):
                        results[0][key] = value
            
            return results[0]
        
    except Exception as e:
        print(f"Error testing seg checkpoint: {e}")
    
    return {}


def test_depth_checkpoint(checkpoint_path):
    """Test Depth checkpoint and return metrics with accurate FPS measurement"""
    try:
        import torch
        import time
        import numpy as np
        from train_depth_only import LightningDepthSegformer, build_depth_datasets
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        
        print(f"\n🔍 Testing Depth checkpoint: {checkpoint_path.name}")
        
        # Load model
        model = LightningDepthSegformer.load_from_checkpoint(
            str(checkpoint_path), 
            strict=False
        )
        
        # Load test dataset
        dataset_root = Path("../dataset")
        _, _, test_ds = build_depth_datasets(dataset_root, "mit_b2", (512, 512))
        
        test_loader = DataLoader(
            test_ds, batch_size=4, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        # Test with PyTorch Lightning (for other metrics)
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        results = trainer.test(model, test_loader, verbose=False)
        
        if results and len(results) > 0:
            # Override FPS with accurate measurement
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Measure FPS accurately
            times = []
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(test_loader):
                    if batch_idx >= 6:  # Measure 6 batches
                        break
                    images = images.to(device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    depth_pred = model(images)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    
                    if batch_idx >= 2:  # Skip warm-up
                        times.append(dt)
            
            if times:
                avg_time = np.mean(times)
                accurate_fps = test_loader.batch_size / avg_time
                results[0]["test_fps"] = accurate_fps
                print(f"   → Accurate FPS: {accurate_fps:.1f}")
            
            # Add validation metrics from log file if available
            log_file = checkpoint_path.parent / "training.log"
            if log_file.exists():
                log_metrics = extract_test_metrics(log_file)
                # Add validation metrics to results
                for key, value in log_metrics.items():
                    if key.startswith("val_"):
                        results[0][key] = value
            
            return results[0]
        
    except Exception as e:
        print(f"Error testing depth checkpoint: {e}")
    
    return {}


def test_mtl_checkpoint(checkpoint_path):
    """Test MTL checkpoint and return metrics with accurate FPS measurement"""
    try:
        import torch
        import time
        import numpy as np
        from train_mtl_segformer import LightningMTLSegformer, build_mtl_datasets
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        
        print(f"\n🔍 Testing MTL checkpoint: {checkpoint_path.name}")
        
        # Load model (strict=False to handle new parameters in updated code)
        model = LightningMTLSegformer.load_from_checkpoint(
            str(checkpoint_path), 
            strict=False
        )
        
        # Load test dataset
        dataset_root = Path("../dataset")
        _, _, test_ds = build_mtl_datasets(dataset_root, "mit_b2", (512, 512))
        
        test_loader = DataLoader(
            test_ds, batch_size=4, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        # Test with PyTorch Lightning (for other metrics)
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        results = trainer.test(model, test_loader, verbose=False)
        
        if results and len(results) > 0:
            # Override FPS with accurate measurement
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Measure FPS accurately
            times = []
            with torch.no_grad():
                for batch_idx, (images, _, _) in enumerate(test_loader):
                    if batch_idx >= 6:  # Measure 6 batches
                        break
                    images = images.to(device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    seg_logits, depth_pred = model(images)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    
                    if batch_idx >= 2:  # Skip warm-up
                        times.append(dt)
            
            if times:
                avg_time = np.mean(times)
                accurate_fps = test_loader.batch_size / avg_time
                results[0]["test_fps"] = accurate_fps
                print(f"   → Accurate FPS: {accurate_fps:.1f}")
            
            # Add validation metrics from log file if available
            log_file = checkpoint_path.parent / "training.log"
            if log_file.exists():
                log_metrics = extract_test_metrics(log_file)
                # Add validation metrics to results
                for key, value in log_metrics.items():
                    if key.startswith("val_"):
                        results[0][key] = value
            
            return results[0]
        
    except Exception as e:
        print(f"Error testing checkpoint: {e}")
    
    return {}


def get_best_checkpoint_by_metric(exp_dir, metric_name, mode="min"):
    """Find best checkpoint by metric (miou, abs_rel, etc.)"""
    if not exp_dir.exists():
        return None
    
    # Find all checkpoints with the metric
    pattern = f"*{metric_name}*.ckpt"
    ckpts = sorted(exp_dir.glob(pattern))
    
    if not ckpts:
        # Fallback to any checkpoint
        ckpts = sorted(exp_dir.glob("*.ckpt"))
        if ckpts:
            return ckpts[0]
        return None
    
    best_ckpt = None
    best_score = 999.0 if mode == "min" else 0.0
    
    for ckpt in ckpts:
        # Extract metric value from filename
        match = re.search(f'{metric_name}=(\d+\.\d+)', ckpt.name)
        if match:
            score = float(match.group(1))
            if (mode == "min" and score < best_score) or (mode == "max" and score > best_score):
                best_score = score
                best_ckpt = ckpt
    
    return best_ckpt if best_ckpt else ckpts[0]


def get_best_mtl_checkpoint(mtl_dir):
    """Find best MTL checkpoint"""
    if not mtl_dir.exists():
        return None
    
    # Find all checkpoints
    absrel_ckpts = sorted(mtl_dir.glob("*val_abs_rel*.ckpt"))
    
    if not absrel_ckpts:
        return None
    
    # Get best AbsRel checkpoint
    best_ckpt = None
    best_absrel = 999.0
    
    for ckpt in absrel_ckpts:
        match = re.search(r'val_abs_rel=(\d+\.\d+)', ckpt.name)
        if match:
            absrel = float(match.group(1))
            if absrel < best_absrel:
                best_absrel = absrel
                best_ckpt = ckpt
    
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Compare Single-Task vs Multi-Task Learning")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                       help="Directory containing experiments")
    parser.add_argument("--seg-version", type=str, default="latest",
                       help="Segmentation experiment version (latest or specific name)")
    parser.add_argument("--depth-version", type=str, default="latest", 
                       help="Depth experiment version (latest or specific name)")
    parser.add_argument("--mtl-version", type=str, default="latest",
                       help="MTL experiment version (latest or specific name)")
    parser.add_argument("--output", type=str, default="single_vs_mtl_comparison.png",
                       help="Output visualization file")
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return
    
    print("\n" + "=" * 80)
    print("SINGLE-TASK vs MULTI-TASK LEARNING COMPARISON")
    print("=" * 80)
    
    # Find experiments
    seg_exps, depth_exps, mtl_exps = find_experiments_by_type(experiments_dir)
    
    print(f"\n📁 Found experiments:")
    print(f"   Seg: {len(seg_exps)} experiments")
    print(f"   Depth: {len(depth_exps)} experiments") 
    print(f"   MTL: {len(mtl_exps)} experiments")
    
    # Select experiments
    results = {}
    
    # Segmentation
    if seg_exps:
        if args.seg_version == "latest":
            seg_exp = seg_exps[0]
        else:
            seg_exp = next((exp for exp in seg_exps if exp.name == args.seg_version), None)
            if not seg_exp:
                print(f"❌ Segmentation experiment '{args.seg_version}' not found!")
                return
        
        print(f"\n🔍 Using Seg experiment: {seg_exp.name}")
        
        # Try to find best checkpoint and test it
        seg_ckpt = get_best_checkpoint_by_metric(seg_exp, "val_miou", "max")
        if seg_ckpt:
            print(f"   → Using checkpoint: {seg_ckpt.name}")
            seg_metrics = test_seg_checkpoint(seg_ckpt)
            if seg_metrics:
                results["Seg-Only"] = seg_metrics
            else:
                results["Seg-Only"] = {}
        else:
            print("   → No checkpoint found, using log file")
            if (seg_exp / "training.log").exists():
                results["Seg-Only"] = extract_test_metrics(seg_exp / "training.log")
            else:
                results["Seg-Only"] = {}
    else:
        results["Seg-Only"] = {}
    
    # Depth
    if depth_exps:
        if args.depth_version == "latest":
            depth_exp = depth_exps[0]
        else:
            depth_exp = next((exp for exp in depth_exps if exp.name == args.depth_version), None)
            if not depth_exp:
                print(f"❌ Depth experiment '{args.depth_version}' not found!")
                return
        
        print(f"🔍 Using Depth experiment: {depth_exp.name}")
        
        # Try to find best checkpoint and test it
        depth_ckpt = get_best_checkpoint_by_metric(depth_exp, "val_abs_rel", "min")
        if depth_ckpt:
            print(f"   → Using checkpoint: {depth_ckpt.name}")
            depth_metrics = test_depth_checkpoint(depth_ckpt)
            if depth_metrics:
                results["Depth-Only"] = depth_metrics
            else:
                results["Depth-Only"] = {}
        else:
            print("   → No checkpoint found, using log file")
            if (depth_exp / "training.log").exists():
                results["Depth-Only"] = extract_test_metrics(depth_exp / "training.log")
            else:
                results["Depth-Only"] = {}
    else:
        results["Depth-Only"] = {}
    
    # MTL
    if mtl_exps:
        if args.mtl_version == "latest":
            mtl_exp = mtl_exps[0]
        else:
            mtl_exp = next((exp for exp in mtl_exps if exp.name == args.mtl_version), None)
            if not mtl_exp:
                print(f"❌ MTL experiment '{args.mtl_version}' not found!")
                return
        
        print(f"🔍 Using MTL experiment: {mtl_exp.name}")
        
        # Try to find best checkpoint
        best_ckpt = get_best_mtl_checkpoint(mtl_exp)
        if best_ckpt:
            print(f"   → Using checkpoint: {best_ckpt.name}")
            mtl_metrics = test_mtl_checkpoint(best_ckpt)
            if mtl_metrics:
                results["MTL"] = mtl_metrics
                # Extract epoch from checkpoint
                epoch_match = re.search(r'epoch=(\d+)', best_ckpt.name)
                if epoch_match:
                    results["MTL"]["best_epoch"] = int(epoch_match.group(1))
                
                # Add validation class IoU from log file for visualization
                if (mtl_exp / "training.log").exists():
                    log_metrics = extract_test_metrics(mtl_exp / "training.log")
                    for key, value in log_metrics.items():
                        if key.startswith("val_class_iou_"):
                            results["MTL"][key] = value
            else:
                results["MTL"] = {}
        else:
            print("   → No checkpoint found, using log file")
            if (mtl_exp / "training.log").exists():
                results["MTL"] = extract_test_metrics(mtl_exp / "training.log")
            else:
                results["MTL"] = {}
    else:
        results["MTL"] = {}
    
    # Sequential Benchmark (if we have both seg and depth experiments)
    sequential_fps = 0
    if seg_exps and depth_exps:
        print(f"\n🔄 Running Sequential Benchmark...")
        try:
            import torch
            from train_seg_only import build_seg_datasets
            from torch.utils.data import DataLoader
            
            # Get latest experiments
            seg_exp = seg_exps[-1]
            depth_exp = depth_exps[-1]
            
            # Find best checkpoints by performance metrics
            seg_ckpt = get_best_checkpoint_by_metric(seg_exp, "val_miou", "max")
            depth_ckpt = get_best_checkpoint_by_metric(depth_exp, "val_abs_rel", "min")
            
            if seg_ckpt and depth_ckpt:
                
                # Load test dataset
                dataset_root = Path("../dataset")
                _, _, test_ds = build_seg_datasets(dataset_root, "mit_b2", (512, 512))
                test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sequential_fps = benchmark_sequential_models(seg_ckpt, depth_ckpt, test_loader, device)
                
                if sequential_fps > 0:
                    results["Sequential"] = {"test_fps": sequential_fps}
                    print(f"   → Sequential FPS: {sequential_fps:.1f}")
        except Exception as e:
            print(f"   → Sequential benchmark failed: {e}")
            sequential_fps = 0
    
    # Check availability
    available = [k for k, v in results.items() if v]
    print(f"\n✅ Available results: {', '.join(available)}")
    
    if len(available) < 2:
        print("\n⚠️  Need at least 2 completed experiments for comparison")
        return
    
    # ========================================================================
    # PERFORMANCE COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Segmentation comparison (use validation mIoU for fair comparison)
    seg_methods = []
    seg_miou = []
    seg_acc = []
    seg_fps = []
    
    for method in ["Seg-Only", "MTL"]:
        if method in results:
            # Use validation mIoU if available, otherwise test mIoU
            miou_key = "val_miou" if "val_miou" in results[method] else "test_miou"
            acc_key = "val_acc" if "val_acc" in results[method] else "test_acc"
            
            if miou_key in results[method]:
                seg_methods.append(method)
                seg_miou.append(results[method].get(miou_key, 0))
                seg_acc.append(results[method].get(acc_key, 0))
                seg_fps.append(results[method].get("test_fps", 0))
    
    if len(seg_methods) == 2:
        print(f"\n{'Method':<15} {'mIoU':<12} {'Accuracy':<12} {'FPS':<12}")
        print("-" * 60)
        
        for i, method in enumerate(seg_methods):
            print(f"{method:<15} {seg_miou[i]:<12.4f} {seg_acc[i]:<12.4f} {seg_fps[i]:<12.1f}")
        
        improvement = ((seg_miou[1] - seg_miou[0]) / seg_miou[0]) * 100
        print(f"\nSegmentation: MTL vs Seg-Only = {improvement:+.2f}%")
    
    # Depth comparison (use validation AbsRel for fair comparison)
    depth_methods = []
    depth_absrel = []
    depth_sqrel = []
    depth_rmse = []
    depth_rmse_log = []
    depth_delta1 = []
    depth_delta2 = []
    depth_delta3 = []
    depth_fps = []
    
    for method in ["Depth-Only", "MTL"]:
        if method in results:
            # Use validation AbsRel if available, otherwise test AbsRel
            absrel_key = "val_abs_rel" if "val_abs_rel" in results[method] else "test_abs_rel"
            sqrel_key = "val_sq_rel" if "val_sq_rel" in results[method] else "test_sq_rel"
            
            if absrel_key in results[method]:
                depth_methods.append(method)
                depth_absrel.append(results[method].get(absrel_key, 999))
                depth_sqrel.append(results[method].get(sqrel_key, 999))
                depth_rmse.append(results[method].get("test_rmse", 999))
                depth_rmse_log.append(results[method].get("test_rmse_log", 999))
                depth_delta1.append(results[method].get("test_delta1", 0))
                depth_delta2.append(results[method].get("test_delta2", 0))
                depth_delta3.append(results[method].get("test_delta3", 0))
                depth_fps.append(results[method].get("test_fps", 0))
    
    if len(depth_methods) == 2:
        print(f"\n{'Method':<15} {'AbsRel':<10} {'SqRel':<10} {'RMSE':<10} {'RMSElog':<10} {'δ<1.25':<10} {'δ<1.25²':<10} {'δ<1.25³':<10} {'FPS':<10}")
        print("-" * 100)
        
        for i, method in enumerate(depth_methods):
            print(f"{method:<15} {depth_absrel[i]:<10.4f} {depth_sqrel[i]:<10.4f} {depth_rmse[i]:<10.4f} {depth_rmse_log[i]:<10.4f} {depth_delta1[i]:<10.4f} {depth_delta2[i]:<10.4f} {depth_delta3[i]:<10.4f} {depth_fps[i]:<10.1f}")
        
        improvement = ((depth_absrel[0] - depth_absrel[1]) / depth_absrel[0]) * 100
        print(f"\nDepth: MTL vs Depth-Only = {improvement:+.2f}% (negative is better)")
    
    # Efficiency comparison
    all_methods = []
    all_fps = []
    
    for method in ["Seg-Only", "Depth-Only", "MTL"]:
        if method in results and "test_fps" in results[method]:
            all_methods.append(method)
            all_fps.append(results[method].get("test_fps", 0))
    
    if len(all_methods) >= 2:
        print(f"\n{'Method':<15} {'FPS':<12} {'Efficiency'}")
        print("-" * 50)
        
        for i, method in enumerate(all_methods):
            print(f"{method:<15} {all_fps[i]:<12.1f}")
        
        # Calculate MTL efficiency vs running separately
        if "Sequential" in results and "MTL" in results:
            # Use actual sequential benchmark results
            sequential_fps_val = results["Sequential"].get("test_fps", 0)
            mtl_fps_val = results["MTL"].get("test_fps", 0)
            
            if sequential_fps_val > 0 and mtl_fps_val > 0:
                speedup = mtl_fps_val / sequential_fps_val
                
                print(f"\nEfficiency Analysis (Real Benchmark):")
                print(f"  Sequential (Seg + Depth): {sequential_fps_val:.1f} FPS")
                print(f"  MTL (both at once):      {mtl_fps_val:.1f} FPS")
                
                if speedup > 1:
                    print(f"  → MTL is {speedup:.2f}x faster! ⚡")
                else:
                    slowdown = 1 / speedup
                    print(f"  → MTL is {slowdown:.2f}x slower 🐌")
                    print(f"  → Sequential is {speedup:.2f}x faster than MTL")
        elif "Seg-Only" in all_methods and "Depth-Only" in all_methods and "MTL" in all_methods:
            # Fallback to theoretical calculation
            seg_idx = all_methods.index("Seg-Only")
            depth_idx = all_methods.index("Depth-Only")
            mtl_idx = all_methods.index("MTL")
            
            seg_fps_val = all_fps[seg_idx]
            depth_fps_val = all_fps[depth_idx]
            mtl_fps_val = all_fps[mtl_idx]
            
            if seg_fps_val > 0 and depth_fps_val > 0 and mtl_fps_val > 0:
                # Calculate combined FPS for running separately
                combined_fps = 1 / (1/seg_fps_val + 1/depth_fps_val)
                speedup = mtl_fps_val / combined_fps
                
                print(f"\nEfficiency Analysis (Theoretical):")
                print(f"  Running separately: {combined_fps:.1f} FPS")
                print(f"  MTL (both at once): {mtl_fps_val:.1f} FPS")
                
                if speedup > 1:
                    print(f"  → MTL is {speedup:.2f}x faster! ⚡")
                else:
                    slowdown = 1 / speedup
                    print(f"  → MTL is {slowdown:.2f}x slower 🐌")
                    print(f"  → Separate is {speedup:.2f}x faster than MTL")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    if seg_methods and depth_methods:
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
        if "Seg-Only" in results and "MTL" in results:
            ax = axes[1, 0]
            class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
            seg_class_iou = []
            mtl_class_iou = []
            
            for class_name in class_names:
                seg_key = f"val_class_iou_{class_name}"
                mtl_key = f"val_class_iou_{class_name}"
                seg_val = results["Seg-Only"].get(seg_key, 0)
                mtl_val = results["MTL"].get(mtl_key, 0)
                seg_class_iou.append(seg_val)
                mtl_class_iou.append(mtl_val)
            
            x = np.arange(len(class_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, seg_class_iou, width, label='Seg-Only', color='#3498db')
            bars2 = ax.bar(x + width/2, mtl_class_iou, width, label='MTL', color='#e74c3c')
            
            ax.set_ylabel('IoU (higher is better)', fontsize=12, fontweight='bold')
            ax.set_title('Class-wise IoU Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Efficiency Analysis
        if "Sequential" in results and "MTL" in results:
            # Use actual sequential benchmark results
            ax = axes[1, 1]
            
            sequential_fps_val = results["Sequential"].get("test_fps", 0)
            mtl_fps_val = results["MTL"].get("test_fps", 0)
            
            if sequential_fps_val > 0 and mtl_fps_val > 0:
                methods = ['Sequential\n(Seg+Depth)', 'MTL\n(Both at once)']
                fps_values = [sequential_fps_val, mtl_fps_val]
                colors = ['#f39c12', '#e74c3c']
                
                x = np.arange(len(methods))
                bars = ax.bar(x, fps_values, color=colors, width=0.6)
                ax.set_ylabel('Effective FPS', fontsize=12, fontweight='bold')
                ax.set_title('Efficiency Comparison (Real Benchmark)', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(methods, fontsize=11)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels and speedup
                for i, (bar, val) in enumerate(zip(bars, fps_values)):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                speedup = mtl_fps_val / sequential_fps_val
                
                if speedup > 1:
                    text = f'Speedup: {speedup:.2f}x'
                    color = "lightgreen"
                else:
                    slowdown = 1 / speedup
                    text = f'Slowdown: {slowdown:.2f}x'
                    color = "lightcoral"
                
                ax.text(0.5, 0.95, text, 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=12, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        elif "Seg-Only" in all_methods and "Depth-Only" in all_methods and "MTL" in all_methods:
            # Fallback to theoretical calculation
            ax = axes[1, 1]
            
            # Calculate combined FPS for running separately
            seg_idx = all_methods.index("Seg-Only")
            depth_idx = all_methods.index("Depth-Only")
            mtl_idx = all_methods.index("MTL")
            
            seg_fps_val = all_fps[seg_idx]
            depth_fps_val = all_fps[depth_idx]
            mtl_fps_val = all_fps[mtl_idx]
            
            if seg_fps_val > 0 and depth_fps_val > 0 and mtl_fps_val > 0:
                combined_fps = 1 / (1/seg_fps_val + 1/depth_fps_val)
                
                methods = ['Separate\n(Seg+Depth)', 'MTL\n(Both at once)']
                fps_values = [combined_fps, mtl_fps_val]
                colors = ['#f39c12', '#e74c3c']
                
                x = np.arange(len(methods))
                bars = ax.bar(x, fps_values, color=colors, width=0.6)
                ax.set_ylabel('Effective FPS', fontsize=12, fontweight='bold')
                ax.set_title('Efficiency Comparison (Theoretical)', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(methods, fontsize=11)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels and speedup
                for i, (bar, val) in enumerate(zip(bars, fps_values)):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                speedup = mtl_fps_val / combined_fps
                
                if speedup > 1:
                    text = f'Speedup: {speedup:.2f}x'
                    color = "lightgreen"
                else:
                    slowdown = 1 / speedup
                    text = f'Slowdown: {slowdown:.2f}x'
                    color = "lightcoral"
                
                ax.text(0.5, 0.95, text, 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=12, fontweight='bold', 
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
