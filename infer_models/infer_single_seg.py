'''python3 /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/infer_models/infer_single_seg.py \
  --ckpt /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/experiments/experiments_seg/reset/rgb-seg-miou-epoch=53-val_miou=0.7569.ckpt \
  --rgb /home/shinds/my_document/DLFromScratch5/test/vae/sss/dataset/test/images/20250526_rfv4_frame_000298_00m_09s.jpg\
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

# 내부 모듈에서 Lightning 모듈 가져오기
from models.train_esanet_seg_only import LightningESANetSegOnly, get_preprocessing_params


def _load_rgb(rgb_path: Path, width: int, height: int):
    

    mean, std = get_preprocessing_params("esanet")

    # RGB
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = TF.resize(rgb, [height, width], interpolation=InterpolationMode.BILINEAR)
    rgb = TF.to_tensor(rgb).contiguous()
    rgb = TF.normalize(rgb, mean=mean, std=std)

    # Shapes: rgb [3,H,W]
    return rgb


@torch.no_grad()
def run_inference(ckpt_path: str, rgb_path: str,
                  device: str = "cuda" if torch.cuda.is_available() else "cpu",
                  height: int = 512, width: int = 512,
                  warmup: int = 10) -> None:
    torch.backends.cudnn.benchmark = True
    # 체크포인트에서 Lightning 모듈 로드 (하이퍼파라미터 포함)
    model: LightningESANetSegOnly = LightningESANetSegOnly.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # 입력 로드 및 전처리 (RGB만)
    rgb = _load_rgb(Path(rgb_path), width=width, height=height)
    rgb = rgb.unsqueeze(0).to(device)      # [1,3,H,W]

    # Warm-up
    for _ in range(max(0, warmup)):
        _ = model(rgb)
    torch.cuda.synchronize()

    # Timing (순수 추론만) - 여러 번 실행하여 통계적 측정
    n_runs = 100
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(rgb)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)
    fps_mean = 1.0 / mean_t
    fps_std = fps_mean * (std_t / mean_t)
    print(f"Mean latency: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms | FPS: {fps_mean:.2f} ± {fps_std:.2f}")
    return fps_mean, fps_std
    


def main():
    parser = argparse.ArgumentParser(description="Infer single RGB image with ESANet RGB-Only Segmentation checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--height", type=int, default=512, help="Model input height")
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations before timing")
    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        rgb_path=args.rgb,
        device=args.device,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
