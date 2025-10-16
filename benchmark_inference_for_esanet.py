#!/usr/bin/env python3
"""
ESANet batch inference benchmark script
"""
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

from train_esanet_mtl import LightningESANetMTL


def benchmark_single_image(model, rgb_path, depth_path, image_size, device="cuda", num_runs=10):
    """Benchmark single image inference"""
    # Load and preprocess RGB image
    rgb_image = Image.open(rgb_path).convert("RGB")
    rgb_image = TF.resize(rgb_image, [image_size[1], image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
    rgb_tensor = TF.to_tensor(rgb_image)
    
    # Load and preprocess depth image
    depth_image = Image.open(depth_path).convert("L")
    depth_image = TF.resize(depth_image, [image_size[1], image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
    depth_tensor = TF.to_tensor(depth_image)
    
    # Get preprocessing params
    try:
        params = smp.encoders.get_preprocessing_params("resnet34")
        mean = params["mean"]
        std = params["std"]
    except:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    rgb_tensor = TF.normalize(rgb_tensor, mean=mean, std=std)
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    depth_tensor = depth_tensor.unsqueeze(0).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(rgb_tensor, depth_tensor)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available() and device == "cuda":
            # GPU-accurate timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                seg_logits, depth_pred = model(rgb_tensor, depth_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            # CPU timing
            t0 = time.perf_counter()
            with torch.no_grad():
                seg_logits, depth_pred = model(rgb_tensor, depth_tensor)
            inference_time = time.perf_counter() - t0
        
        times.append(inference_time)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "fps": 1.0 / np.mean(times)
    }


def benchmark_batch(model, rgb_paths, depth_paths, image_size, device="cuda", batch_sizes=[1, 2, 4, 8]):
    """Benchmark batch inference"""
    results = {}
    
    for batch_size in batch_sizes:
        if len(rgb_paths) < batch_size:
            continue
            
        # Prepare batch
        batch_rgb = []
        batch_depth = []
        for i in range(batch_size):
            # RGB image
            rgb_image = Image.open(rgb_paths[i]).convert("RGB")
            rgb_image = TF.resize(rgb_image, [image_size[1], image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
            rgb_tensor = TF.to_tensor(rgb_image)
            
            # Depth image
            depth_image = Image.open(depth_paths[i]).convert("L")
            depth_image = TF.resize(depth_image, [image_size[1], image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
            depth_tensor = TF.to_tensor(depth_image)
            
            try:
                params = smp.encoders.get_preprocessing_params("resnet34")
                mean = params["mean"]
                std = params["std"]
            except:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            
            rgb_tensor = TF.normalize(rgb_tensor, mean=mean, std=std)
            batch_rgb.append(rgb_tensor)
            batch_depth.append(depth_tensor)
        
        rgb_batch = torch.stack(batch_rgb).to(device)
        depth_batch = torch.stack(batch_depth).to(device)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(rgb_batch, depth_batch)
        
        # Benchmark
        times = []
        for _ in range(10):
            if torch.cuda.is_available() and device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.no_grad():
                    seg_logits, depth_pred = model(rgb_batch, depth_batch)
                end_event.record()
                
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    seg_logits, depth_pred = model(rgb_batch, depth_batch)
                inference_time = time.perf_counter() - t0
            
            times.append(inference_time)
        
        results[batch_size] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "fps": batch_size / np.mean(times),
            "time_per_image": np.mean(times) / batch_size
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ESANet MTL model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--rgb-dir", type=str, required=True, help="Directory containing RGB test images")
    parser.add_argument("--depth-dir", type=str, required=True, help="Directory containing depth test images")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder name")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], help="Image size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for averaging")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to test")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = LightningESANetMTL.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)
    print("Model loaded successfully")
    
    # Get test images
    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    
    rgb_paths = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    depth_paths = sorted(list(depth_dir.glob("*.jpg")) + list(depth_dir.glob("*.png")))
    
    if not rgb_paths or not depth_paths:
        print(f"No images found in {rgb_dir} or {depth_dir}")
        return
    
    print(f"Found {len(rgb_paths)} RGB images and {len(depth_paths)} depth images")
    
    # Single image benchmark
    print("\n" + "="*60)
    print("SINGLE IMAGE BENCHMARK")
    print("="*60)
    
    single_results = benchmark_single_image(
        model, rgb_paths[0], depth_paths[0], args.image_size, args.device, args.num_runs
    )
    
    print(f"Mean inference time: {single_results['mean_time']:.4f} ± {single_results['std_time']:.4f} seconds")
    print(f"Min time: {single_results['min_time']:.4f} seconds")
    print(f"Max time: {single_results['max_time']:.4f} seconds")
    print(f"FPS: {single_results['fps']:.2f}")
    print(f"Latency: {single_results['mean_time']*1000:.2f} ms")
    
    # Batch benchmark
    print("\n" + "="*60)
    print("BATCH BENCHMARK")
    print("="*60)
    
    batch_results = benchmark_batch(
        model, rgb_paths, depth_paths, args.image_size, args.device, args.batch_sizes
    )
    
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'FPS':<12} {'Latency (ms)':<15} {'Time/Image (s)':<15}")
    print("-" * 75)
    
    for batch_size, results in batch_results.items():
        latency_ms = results['mean_time'] * 1000
        print(f"{batch_size:<12} {results['mean_time']:.4f}±{results['std_time']:.4f} {results['fps']:.2f} {latency_ms:.2f} {results['time_per_image']:.4f}")


if __name__ == "__main__":
    main()
