'''python3 /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/infer_models/infer_single_mtl.py \
  --ckpt /home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/experiments/experiments_mtl/esanet_ppm_revise_dwa2/esanet-mtl-absrel-epoch=186-val_abs_rel=0.0704.ckpt \
  --rgb /home/shinds/my_document/DLFromScratch5/test/vae/sss/dataset/test/images/20250526_rfv4_frame_000298_00m_09s.jpg \
  --depth /home/shinds/my_document/DLFromScratch5/test/vae/sss/dataset/test/depth/20250526_rfv4_frame_000298_00m_09s_depth.png \
  --height 512 --width 512 --warmup 100'''


import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

# 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈에서 Lightning 모듈 가져오기
from models.train_esanet_mtl_dwa import LightningESANetMTL, get_preprocessing_params


def _load_rgb_depth(rgb_path: Path, depth_path: Path, width: int, height: int):
    

    mean, std = get_preprocessing_params("esanet")

    # RGB
    rgb = Image.open(rgb_path).convert("RGB")
    rgb = TF.resize(rgb, [height, width], interpolation=InterpolationMode.BILINEAR)
    rgb = TF.to_tensor(rgb).contiguous()
    rgb = TF.normalize(rgb, mean=mean, std=std)

    # Depth
    dimg = Image.open(depth_path)
    if dimg.mode == 'I;16':
        depth_np = np.array(dimg, dtype=np.float32) / 65535.0
    elif dimg.mode == 'I':
        depth_np = np.array(dimg, dtype=np.float32)
        if depth_np.max() > 1.0:
            depth_np = depth_np / max(1.0, float(depth_np.max()))
    else:
        depth_np = np.array(dimg.convert('L'), dtype=np.float32) / 255.0
    depth = Image.fromarray(depth_np)
    depth = TF.resize(depth, [height, width], interpolation=InterpolationMode.BILINEAR)
    depth = torch.from_numpy(np.array(depth, dtype=np.float32)).contiguous().unsqueeze(0)

    # Shapes: rgb [3,H,W], depth [1,H,W]
    return rgb, depth


@torch.no_grad()
def run_inference(ckpt_path: str, rgb_path: str, depth_path: str,
                  device: str = "cuda" if torch.cuda.is_available() else "cpu",
                  height: int = 512, width: int = 512,
                  warmup: int = 10) -> None:
    # CuDNN 최적화 활성화 (고정 입력 크기에 최적화)
    torch.backends.cudnn.benchmark = True

    # 체크포인트에서 Lightning 모듈 로드 (하이퍼파라미터 포함)
    model: LightningESANetMTL = LightningESANetMTL.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # 입력 로드 및 전처리
    rgb, depth = _load_rgb_depth(Path(rgb_path), Path(depth_path), width=width, height=height)
    rgb = rgb.unsqueeze(0).to(device)      # [1,3,H,W]
    depth = depth.unsqueeze(0).to(device)  # [1,1,H,W]

    # Warm-up
    for _ in range(max(0, warmup)):
        _ = model(rgb, depth)
    torch.cuda.synchronize()  # warm-up 후 동기화

    # Timing (순수 추론만) - 여러 번 실행하여 통계적 측정
    n_runs = 100
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(rgb, depth)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)
    
    fps_mean = 1.0 / mean_t
    fps_std = fps_mean * (std_t / mean_t)
    print(f"Mean latency: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms | FPS: {fps_mean:.2f} ± {fps_std:.2f}")
    


def main():
    parser = argparse.ArgumentParser(description="Infer single RGB+Depth image with ESANet MTL checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth image")
    
    parser.add_argument("--height", type=int, default=512, help="Model input height")
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations before timing")
    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        rgb_path=args.rgb,
        depth_path=args.depth,
        device=args.device,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()


