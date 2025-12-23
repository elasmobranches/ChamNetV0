#!/usr/bin/env python3
"""
Perfect MTL Inference - 100% Working!
- 실제 모델 추론 (더미 아님!)
- GT와 Prediction 비교
- 2행 3열 시각화
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# MMSeg imports - 순서 중요!
sys.path.insert(0, os.getcwd())

# Registry 초기화를 위해 mmseg import
import mmseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# 모든 모듈 등록 (이게 핵심!)
register_all_modules(init_default_scope=True)
print("✓ All modules registered")

# 기타 필요한 imports
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. 논문용 전역 폰트 설정 (수정됨)
plt.rcParams.update({
    'font.size': 12,
    # 구체적인 폰트 이름 하나만 지정하는 대신, 우선순위 리스트를 줍니다.
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.weight': 'normal',
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'image.cmap': 'viridis'
})

# 클래스 정의 (7 classes)
CLASSES = ['background', 'chamoe', 'heatpipe', 'path', 'pillar', 'topdownfarm', 'unknown']

COLOR_MAP = {
    0: (0, 0, 0),        # background - black
    1: (255, 255, 0),    # chamoe - yellow
    2: (255, 0, 0),      # heatpipe - red
    3: (0, 255, 0),      # path - green
    4: (0, 0, 255),      # pillar - blue
    5: (255, 0, 255),    # topdownfarm - magenta
    6: (128, 128, 128),  # unknown - gray
}

def build_mtl_model(ckpt_path, device='cuda'):
    """MTL 모델 빌드 및 체크포인트 로드"""

    print("🔨 Building MTL model...")

    # MTL Config 프로그래밍 방식으로 생성
    norm_cfg = dict(type='SyncBN', requires_grad=True)

    cfg_dict = dict(
        type='MTLEncoderDecoder',
        data_preprocessor=dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
            size=(512, 512)
        ),
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        ),
        seg_decode_head=dict(
            type='SegformerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        depth_decode_head=dict(
            type='DepthHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='SILogLoss', loss_weight=1.0)
        ),
        mtl_config=dict(
            weight_strategy='manual',
            fixed_weights=dict(
                seg_weight=1.0,
                depth_weight=5.0
            ),
            uncertainty_config=dict(
                enabled=True,
                init_log_var_seg=0.0,
                init_log_var_depth=0.0
            ),
            dwa_config=dict(
                enabled=False,
                num_tasks=2,
                temperature=2.0,
                update_freq=50,
                window_size=10
            )
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )

    cfg = Config(cfg_dict)

    # 모델 빌드
    try:
        model = MODELS.build(cfg)
        print("✓ Model built successfully")
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        raise

    # 체크포인트 로드
    print(f"📥 Loading checkpoint: {ckpt_path}")
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    print("✓ Checkpoint loaded")

    model.to(device)
    model.eval()

    return model


def load_gt_data(img_path, mask_folder, depth_folder):
    """GT 데이터 로드"""

    # 파일명 추출
    img_name = Path(img_path).stem

    # Mask GT
    mask_path = Path(mask_folder) / f"{img_name}_mask.png"
    if mask_path.exists():
        mask_gt = np.array(Image.open(mask_path))
    else:
        print(f"⚠ Mask GT not found: {mask_path}")
        mask_gt = None

    # Depth GT
    depth_path = Path(depth_folder) / f"{img_name}_depth.npy"
    if depth_path.exists():
        depth_gt = np.load(depth_path)
        # Normalize for visualization
        if depth_gt.max() > depth_gt.min():
            depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min())
    else:
        print(f"⚠ Depth GT not found: {depth_path}")
        depth_gt = None

    return mask_gt, depth_gt


def preprocess_image(img_path, device='cuda'):
    """이미지 전처리"""

    # 이미지 로드
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    # Resize to 512x512
    img_resized = img.resize((512, 512), Image.BILINEAR)
    img_np = np.array(img_resized)

    # Normalize
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_norm = (img_np - mean) / std

    # To tensor: (H,W,C) -> (1,C,H,W)
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # 메타 정보 생성
    img_meta = {
        'img_shape': (512, 512),
        'ori_shape': (original_size[1], original_size[0]),  # (H, W)
        'pad_shape': (512, 512),
        'scale_factor': (512 / original_size[0], 512 / original_size[1]),
        'flip': False,
        'flip_direction': None
    }

    return img_tensor, img_np, original_size, img_meta


def run_inference(model, img_tensor, img_meta, device='cuda'):
    """실제 추론 실행"""
    from mmseg.structures import SegDataSample

    with torch.no_grad():
        # SegDataSample 생성
        data_sample = SegDataSample()
        data_sample.set_metainfo(img_meta)
        
        # MTL 모델 forward
        result = model(img_tensor, data_samples=[data_sample], mode='predict')
        
        # 결과는 리스트로 반환됨
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Segmentation 추출
        if hasattr(result, 'pred_sem_seg'):
            seg_pred = result.pred_sem_seg.data.cpu().numpy()
            if seg_pred.ndim == 3:  # (1, H, W)
                seg_pred = seg_pred[0]
        else:
            seg_pred = None

        # Depth 추출
        if hasattr(result, 'pred_depth_map'):
            depth_pred = result.pred_depth_map.data.cpu().numpy()
            if depth_pred.ndim == 3:  # (1, H, W)
                depth_pred = depth_pred[0]
            
            # Normalize to [0, 1]
            if depth_pred.max() > depth_pred.min():
                depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())
        else:
            depth_pred = None

    return seg_pred, depth_pred


def visualize_2x3_grid(img_np, mask_gt, seg_pred, depth_gt, depth_pred, output_path, img_name=''):
    """
    논문용 High-Quality 2x3 Grid 시각화 (수정버전)
    - int64 -> uint8 형변환 추가 (에러 해결 핵심)
    """
    
    # Figure 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    
    if img_name:
        fig.suptitle(f'MTL Inference Results: {img_name}', fontweight='bold')

    # --- 반복되는 imshow 작업을 위한 헬퍼 함수 ---
    def show_img(ax, img, title, is_mask=False, cmap=None, vmin=None, vmax=None, add_cbar=False):
        if img is None:
            ax.text(0.5, 0.5, 'Not Available', ha='center', va='center')
            ax.axis('off')
            return

        # ==========================================================
        # 🚨 [핵심 수정] 데이터 타입 안전 변환
        # int64(모델출력) -> uint8(이미지) 변환이 없으면 PIL 에러 발생
        # ==========================================================
        if is_mask:
            img = img.astype(np.uint8)  # Mask는 무조건 uint8
        elif img.dtype == np.float64:
            img = img.astype(np.float32) # Depth/RGB가 float64면 float32로
        # ==========================================================

        if is_mask:
            # Mask 리사이징 (Nearest Neighbor - 픽셀값 유지)
            img_pil = Image.fromarray(img).resize((512, 512), Image.NEAREST)
            img_arr = np.array(img_pil)
            
            # Color Mapping
            color_img = np.zeros((512, 512, 3), dtype=np.uint8)
            for label_id, color in COLOR_MAP.items():
                color_img[img_arr == label_id] = color
            im = ax.imshow(color_img)
        else:
            # 일반 이미지/Depth 리사이징 (Bilinear - 부드럽게)
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1): # Depth
                img_pil = Image.fromarray(img).resize((512, 512), Image.BILINEAR)
            else: # RGB
                img_pil = Image.fromarray(img.astype(np.uint8)).resize((512, 512), Image.BILINEAR)
            
            im = ax.imshow(np.array(img_pil), cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.axis('off')

        # Colorbar 추가
        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Depth (Normalized)', rotation=270, labelpad=15)
        
        return im

    # --- Row 1: Segmentation ---
    show_img(axes[0, 0], mask_gt, '(a) Segmentation GT', is_mask=True)
    show_img(axes[0, 1], img_np, '(b) Input RGB')
    show_img(axes[0, 2], seg_pred, '(c) Segmentation Pred', is_mask=True)

    # --- Row 2: Depth ---
    show_img(axes[1, 0], depth_gt, '(d) Depth GT', cmap='plasma', vmin=0, vmax=1, add_cbar=True)
    show_img(axes[1, 1], img_np, '(e) Input RGB')
    show_img(axes[1, 2], depth_pred, '(f) Depth Pred', cmap='plasma', vmin=0, vmax=1, add_cbar=True)

    # --- Legend ---
    if seg_pred is not None:
        unique, counts = np.unique(seg_pred, return_counts=True)
        total = seg_pred.size
        legend_labels = [f"{CLASSES[u]} ({counts[i]/total*100:.1f}%)" for i, u in enumerate(unique)]
        
        text_str = "\n".join(legend_labels)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        axes[0, 2].text(0.95, 0.05, text_str, transform=axes[0, 2].transAxes, 
                        fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def process_folder(ckpt_path, img_folder, mask_folder, depth_folder, output_dir, device='cuda', num_samples=None):
    """폴더 내 모든 이미지 처리"""

    # 출력 디렉토리
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 파일 찾기
    img_exts = {'.jpg', '.jpeg', '.png'}
    img_files = []
    for ext in img_exts:
        img_files.extend(Path(img_folder).glob(f'*{ext}'))
        img_files.extend(Path(img_folder).glob(f'*{ext.upper()}'))

    img_files = sorted(img_files)

    if num_samples:
        img_files = img_files[:num_samples]

    print(f"\n{'='*80}")
    print(f"🎯 PERFECT MTL Inference")
    print(f"{'='*80}")
    print(f"📁 Images: {img_folder} ({len(img_files)} files)")
    print(f"📁 Masks GT: {mask_folder}")
    print(f"📁 Depth GT: {depth_folder}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    print(f"{'='*80}\n")

    if len(img_files) == 0:
        print("❌ No images found!")
        return

    # 모델 빌드
    model = build_mtl_model(ckpt_path, device)

    print(f"\n🚀 Processing {len(img_files)} images...\n")

    success_count = 0
    for img_file in tqdm(img_files, desc="Inference"):
        try:
            # 1. 이미지 전처리
            img_tensor, img_np, original_size, img_meta = preprocess_image(str(img_file), device)

            # 2. GT 로드
            mask_gt, depth_gt = load_gt_data(str(img_file), mask_folder, depth_folder)

            # 3. 추론
            seg_pred, depth_pred = run_inference(model, img_tensor, img_meta, device)

            # 4. 시각화 (2x3 grid)
            output_path = os.path.join(output_dir, f"{img_file.stem}_result.png")
            visualize_2x3_grid(img_np, mask_gt, seg_pred, depth_gt, depth_pred,
                              output_path, img_file.name)

            success_count += 1

        except Exception as e:
            print(f"\n❌ Error: {img_file.name}")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ SUCCESS!")
    print(f"   Processed: {success_count}/{len(img_files)} images")
    print(f"   Saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Perfect MTL Inference')
    parser.add_argument('checkpoint', help='Checkpoint path (.pth)')
    parser.add_argument('--img-folder', required=True, help='Image folder')
    parser.add_argument('--mask-folder', default='dataset/test/masks', help='Mask GT folder')
    parser.add_argument('--depth-folder', default='dataset/test/metric_depth/depth_npy', help='Depth GT folder')
    parser.add_argument('--output-dir', default='./visualizations/perfect2', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--num-samples', type=int, help='Max samples')

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not os.path.exists(args.img_folder):
        print(f"❌ Image folder not found: {args.img_folder}")
        return

    # Process
    process_folder(
        args.checkpoint,
        args.img_folder,
        args.mask_folder,
        args.depth_folder,
        args.output_dir,
        args.device,
        args.num_samples
    )


if __name__ == '__main__':
    main()
