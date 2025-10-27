'''python3 /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/infer_models/infer_sequential_seg_depth.py \
  --seg_ckpt /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/experiments/experiments_seg/1/rgb-seg-miou-epoch=30-val_miou=0.7699.ckpt \
  --depth_ckpt /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/experiments/experiments_depth/1/depth-abs-rel-epoch=47-val_abs_rel=0.1909.ckpt \
  --rgb /home/shinds/my_document/DLFromScratch5/test/vae/sss/dataset/test/images/20250526_rfv4_frame_000298_00m_09s.jpg \
  --height 512 --width 512 --warmup 100'''


import argparse
import os
import statistics
import time
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

# 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train_esanet_seg_only import LightningESANetSegOnly, get_preprocessing_params
from models.train_esanet_depth_only import LightningESANetDepthOnly


def _load_rgb(rgb_path: Path, width: int, height: int):
    mean, std = get_preprocessing_params("esanet")
    # RGB 이미지 로드
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = TF.resize(rgb, [height, width], interpolation=InterpolationMode.BILINEAR)
    rgb = TF.to_tensor(rgb).contiguous()
    rgb = TF.normalize(rgb, mean=mean, std=std)

    # Shapes: rgb [3,H,W]
    return rgb


@torch.no_grad()
def run_sequential_inference(seg_ckpt_path: str, depth_ckpt_path: str, rgb_path: str,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu",
                           height: int = 512, width: int = 512,
                           warmup: int = 10) -> None:
    torch.backends.cudnn.benchmark = True
    
    # Segmentation 모델 로드
    seg_model: LightningESANetSegOnly = LightningESANetSegOnly.load_from_checkpoint(seg_ckpt_path, map_location=device)
    seg_model.eval()
    seg_model.to(device)
    
    # Depth 모델 로드
    depth_model: LightningESANetDepthOnly = LightningESANetDepthOnly.load_from_checkpoint(depth_ckpt_path, map_location=device)
    depth_model.eval()
    depth_model.to(device)

    # 입력 로드 및 전처리 (RGB만)
    rgb = _load_rgb(Path(rgb_path), width=width, height=height)
    rgb = rgb.unsqueeze(0).to(device)      # [1,3,H,W]

    # 각 모델별로 따로 warm-up
    print("Warming up segmentation model...")
    for _ in range(max(0, warmup)):
        _ = seg_model(rgb)
    torch.cuda.synchronize()
    
    print("Warming up depth model...")
    for _ in range(max(0, warmup)):
        _ = depth_model(rgb)
    torch.cuda.synchronize()

    # 개별 모델 성능 측정
    print("Measuring individual model performance...")
    
    # Segmentation 모델만 측정
    seg_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = seg_model(rgb)
        torch.cuda.synchronize()
        seg_times.append(time.perf_counter() - t0)
    
    # Depth 모델만 측정
    depth_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = depth_model(rgb)
        torch.cuda.synchronize()
        depth_times.append(time.perf_counter() - t0)
    
    # 연속 실행 성능 측정
    print("Measuring sequential execution performance...")
    sequential_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = seg_model(rgb)
        _ = depth_model(rgb)
        torch.cuda.synchronize()
        sequential_times.append(time.perf_counter() - t0)

    # 결과 계산 및 출력
    seg_mean = statistics.mean(seg_times)
    seg_std = statistics.stdev(seg_times)
    seg_fps = 1.0 / seg_mean
    seg_fps_std = seg_fps * (seg_std / seg_mean)
    
    depth_mean = statistics.mean(depth_times)
    depth_std = statistics.stdev(depth_times)
    depth_fps = 1.0 / depth_mean
    depth_fps_std = depth_fps * (depth_std / depth_mean)
    
    seq_mean = statistics.mean(sequential_times)
    seq_std = statistics.stdev(sequential_times)
    seq_fps = 1.0 / seq_mean
    seq_fps_std = seq_fps * (seq_std / seq_mean)
    
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    print(f"Segmentation Model Only:")
    print(f"  Latency: {seg_mean*1000:.2f} ± {seg_std*1000:.2f} ms")
    print(f"  FPS: {seg_fps:.2f} ± {seg_fps_std:.2f}")
    print()
    print(f"Depth Model Only:")
    print(f"  Latency: {depth_mean*1000:.2f} ± {depth_std*1000:.2f} ms")
    print(f"  FPS: {depth_fps:.2f} ± {depth_fps_std:.2f}")
    print()
    print(f"Sequential Execution (Seg + Depth):")
    print(f"  Latency: {seq_mean*1000:.2f} ± {seq_std*1000:.2f} ms")
    print(f"  FPS: {seq_fps:.2f} ± {seq_fps_std:.2f}")
    print()
    print(f"Efficiency Analysis:")
    print(f"  Theoretical combined latency: {(seg_mean + depth_mean)*1000:.2f} ms")
    print(f"  Actual sequential latency: {seq_mean*1000:.2f} ms")
    print(f"  Overhead: {(seq_mean - seg_mean - depth_mean)*1000:.2f} ms")
    print("="*60)
    


def main():
    parser = argparse.ArgumentParser(description="Sequential inference with ESANet Segmentation and Depth models")
    parser.add_argument("--seg_ckpt", type=str, required=True, help="Path to segmentation .ckpt checkpoint")
    parser.add_argument("--depth_ckpt", type=str, required=True, help="Path to depth .ckpt checkpoint")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--height", type=int, default=512, help="Model input height")
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations before timing")
    args = parser.parse_args()

    run_sequential_inference(
        seg_ckpt_path=args.seg_ckpt,
        depth_ckpt_path=args.depth_ckpt,
        rgb_path=args.rgb,
        device=args.device,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
