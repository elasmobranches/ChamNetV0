# ============================================================================
# 데이터셋 증강 스크립트 (Albumentations 기반)
# ============================================================================
# 수정사항: RandomScale 제거 (이미지 해상도 일관성 유지)
#
# 주요 기능:
# - RGB와 Depth에 서로 다른 증강 기법 적용
# - 기하학적 변환(회전, 반전)은 RGB/Depth/Mask에 동일하게 적용
# - 색상 증강은 RGB에만 적용
# - 사용자가 지정한 배수만큼 데이터 증강하여 저장

import argparse
import os
import sys
import random
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
import shutil
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed: int):
    """재현성을 위해 모든 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


# ============================================================================
# 상수 정의
# ============================================================================
# 지원되는 이미지 파일 확장자
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 시각화를 위한 컬러 맵 지정 (coco_to_masks.py에서 가져옴)
COLOR_MAP = {
    0: (0, 0, 0),        # background - black
    1: (255, 255, 0),    # chamoe - yellow
    2: (255, 0, 0),      # heatpipe - red
    3: (0, 255, 0),      # path - green
    4: (0, 0, 255),      # pillar - blue
    5: (255, 0, 255),    # topdownfarm - magenta
    6: (128, 128, 128),  # unknown - gray
}

# ============================================================================
# 유틸리티 함수
# ============================================================================
def labels_to_color_image(label_mask: np.ndarray) -> np.ndarray:
    """
    라벨 마스크(HxW, uint8)를 RGB 컬러 이미지(HxWx3, uint8)로 변환.
    
    Args:
        label_mask: 그레이스케일 라벨 마스크
        
    Returns:
        RGB 컬러 이미지
    """
    height, width = label_mask.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value, rgb in COLOR_MAP.items():
        color_img[label_mask == label_value] = rgb
    return color_img


# ============================================================================
# 증강 기법 정의
# ============================================================================
def get_rgb_augmentations():
    """
    RGB 전용 색상 증강 기법을 반환합니다.
    
    Returns:
        A.Compose: RGB용 albumentations 컴포즈
    """
    return A.Compose([
        # 밝기/대비 조정
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.3
        ),
        
        # 가우시안 노이즈 추가
        A.ImageCompression(
            quality_lower=80,
            quality_upper=100,
            p=0.3
        ),

        A.MotionBlur(
            blur_limit=5,
            p=0.2
        ),
        
        # 감마 보정
        A.RandomGamma(
            gamma_limit=(90, 110), 
            p=0.2
        ),
        
        # HSV 색상 공간 변환
        A.HueSaturationValue(
            hue_shift_limit=0, 
            sat_shift_limit=20, 
            val_shift_limit=15, 
            p=0.2
        ),
        
        # 추가 색상 지터
        A.ColorJitter(
            brightness=0.1, 
            contrast=0.1, 
            saturation=0.1, 
            hue=0.00, 
            p=0.2
        ),
    ])


def get_geometric_augmentations():
    """
    기하학적 변환 증강 기법을 반환합니다.
    RGB, Mono Depth, Metric Depth, Mask에 동일하게 적용됩니다.
    
    [수정] RandomScale을 제거하여 출력 이미지 크기를 원본과 동일하게 유지합니다.
    
    Returns:
        A.Compose: 기하학적 변환용 albumentations 컴포즈
    """
    return A.Compose([
        # 좌우 반전 (크기 유지)
        A.HorizontalFlip(p=0.5),
        
        # 회전 (최대 15도) - depth 경계는 0으로 채움 (크기 유지)
        A.Rotate(
            limit=15, 
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        
        # [삭제됨] A.RandomScale(scale_limit=0.1, p=0.2)
        # 이유: Resize 단계 없이 저장하므로 이미지 크기가 변하면 학습 시 배치 구성에 실패함.
        
    ], additional_targets={
        'mono_depth_npy': 'image', 
        'mono_depth_png': 'image', 
        'metric_depth_npy': 'image',
        'metric_depth_png': 'image',
        'mask': 'mask'
    })


# ============================================================================
# 데이터셋 증강 클래스
# ============================================================================
class DatasetAugmentor:
    """
    RGB/Depth 데이터셋을 증강하는 클래스입니다.
    """
    
    def __init__(self, images_dir: Path, masks_dir: Path, mono_depth_dir: Path, 
                 metric_depth_dir: Path, output_root: Path, multiplier: int):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mono_depth_dir = Path(mono_depth_dir)
        self.metric_depth_dir = Path(metric_depth_dir)
        self.output_root = Path(output_root)
        self.multiplier = multiplier
        
        # 증강 기법 초기화
        self.rgb_augment = get_rgb_augmentations()
        self.geometric_augment = get_geometric_augmentations()
        
        # 출력 디렉토리 생성
        self._create_output_directories()
        
    def _create_output_directories(self):
        """출력 디렉토리 구조를 생성합니다."""
        for data_type in ['images', 'masks', 'masks_color']:
            output_dir = self.output_root / 'train' / data_type
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {output_dir}")
        
        for depth_type in ['mono_depth', 'metric_depth']:
            for sub_dir in ['depth_npy', 'depth_visualization']:
                output_dir = self.output_root / 'train' / depth_type / sub_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {output_dir}")
    
    def _get_image_files(self, split: str) -> List[Path]:
        """특정 split의 이미지 파일 목록을 반환합니다."""
        images_dir = self.images_dir
        if not images_dir.exists():
            print(f"⚠️ Warning: {images_dir} does not exist")
            return []
        
        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def _load_image_data(self, image_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """이미지와 관련된 마스크, 깊이 데이터를 로드합니다."""
        img_stem = image_path.stem
        
        mask_path = self.masks_dir / f"{img_stem}_mask.png"
        mono_depth_npy_path = self.mono_depth_dir / "depth_npy" / f"{img_stem}_depth.npy"
        mono_depth_png_path = self.mono_depth_dir / "depth_visualization" / f"{img_stem}_depth.png"
        metric_depth_npy_path = self.metric_depth_dir / "depth_npy" / f"{img_stem}_depth.npy"
        metric_depth_png_path = self.metric_depth_dir / "depth_visualization" / f"{img_stem}_depth.png"
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 로드
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
        else:
            print(f"⚠️ Warning: Mask file not found: {mask_path}")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        target_size = (image.shape[1], image.shape[0])
        
        # Mono Depth Load
        if mono_depth_npy_path.exists():
            mono_depth_npy = np.load(str(mono_depth_npy_path))
            if mono_depth_npy.shape[:2] != image.shape[:2]:
                mono_depth_npy = cv2.resize(mono_depth_npy, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            print(f"⚠️ Warning: Mono depth npy file not found: {mono_depth_npy_path}")
            mono_depth_npy = np.zeros(image.shape[:2], dtype=np.float32)
            
        if mono_depth_png_path.exists():
            mono_depth_png = cv2.imread(str(mono_depth_png_path), cv2.IMREAD_UNCHANGED)
            if len(mono_depth_png.shape) == 3:
                mono_depth_png = mono_depth_png[:, :, 0]
            if mono_depth_png.shape[:2] != image.shape[:2]:
                mono_depth_png = cv2.resize(mono_depth_png, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            print(f"⚠️ Warning: Mono depth png file not found: {mono_depth_png_path}")
            mono_depth_png = np.zeros(image.shape[:2], dtype=np.uint8)
            
        # Metric Depth Load
        if metric_depth_npy_path.exists():
            metric_depth_npy = np.load(str(metric_depth_npy_path))
            if metric_depth_npy.shape[:2] != image.shape[:2]:
                metric_depth_npy = cv2.resize(metric_depth_npy, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            print(f"⚠️ Warning: Metric depth npy file not found: {metric_depth_npy_path}")
            metric_depth_npy = np.zeros(image.shape[:2], dtype=np.float32)
            
        if metric_depth_png_path.exists():
            metric_depth_png = cv2.imread(str(metric_depth_png_path), cv2.IMREAD_UNCHANGED)
            if len(metric_depth_png.shape) == 3:
                metric_depth_png = metric_depth_png[:, :, 0]
            if metric_depth_png.shape[:2] != image.shape[:2]:
                metric_depth_png = cv2.resize(metric_depth_png, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            print(f"⚠️ Warning: Metric depth png file not found: {metric_depth_png_path}")
            metric_depth_png = np.zeros(image.shape[:2], dtype=np.uint8)
        
        return image, mask, mono_depth_npy, mono_depth_png, metric_depth_npy, metric_depth_png
    
    def _save_augmented_data(self, image: np.ndarray, mask: np.ndarray, 
                           mono_depth_npy: np.ndarray, mono_depth_png: np.ndarray,
                           metric_depth_npy: np.ndarray, metric_depth_png: np.ndarray,
                           split: str, original_name: str, aug_idx: int):
        """증강된 데이터를 저장합니다."""
        base_name = f"{original_name}_aug_{aug_idx:03d}"
        
        # RGB 이미지
        image_path = self.output_root / split / 'images' / f"{base_name}.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)
        
        # 마스크
        mask_path = self.output_root / split / 'masks' / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # 컬러 마스크
        color_mask = labels_to_color_image(mask)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        color_mask_path = self.output_root / split / 'masks_color' / f"{base_name}_mask.png"
        cv2.imwrite(str(color_mask_path), color_mask_bgr)
        
        # Mono Depth
        mono_depth_npy_path = self.output_root / split / 'mono_depth' / 'depth_npy' / f"{base_name}_depth.npy"
        np.save(str(mono_depth_npy_path), mono_depth_npy)
        
        mono_depth_png_path = self.output_root / split / 'mono_depth' / 'depth_visualization' / f"{base_name}_depth.png"
        cv2.imwrite(str(mono_depth_png_path), mono_depth_png)
        
        # Metric Depth
        metric_depth_npy_path = self.output_root / split / 'metric_depth' / 'depth_npy' / f"{base_name}_depth.npy"
        np.save(str(metric_depth_npy_path), metric_depth_npy)
        
        metric_depth_png_path = self.output_root / split / 'metric_depth' / 'depth_visualization' / f"{base_name}_depth.png"
        cv2.imwrite(str(metric_depth_png_path), metric_depth_png)
    
    def augment_split(self, split: str):
        """특정 split의 데이터를 증강합니다."""
        print(f"\n🔄 Augmenting {split} split...")
        
        image_files = self._get_image_files(split)
        if not image_files:
            print(f"⚠️ No images found in {split} split")
            return
        
        print(f"📊 Found {len(image_files)} images in {split} split")
        
        for img_idx, image_path in enumerate(tqdm(image_files, desc=f"Augmenting {split}")):
            try:
                # 원본 데이터 로드
                image, mask, mono_depth_npy, mono_depth_png, metric_depth_npy, metric_depth_png = \
                    self._load_image_data(image_path, split)
                original_name = image_path.stem
                
                # 원본 데이터 저장 (aug_000)
                self._save_augmented_data(
                    image, mask, 
                    mono_depth_npy, mono_depth_png, 
                    metric_depth_npy, metric_depth_png,
                    split, original_name, 0
                )
                
                # 증강 수행
                for aug_idx in range(1, self.multiplier):
                    # RGB 색상 증강
                    rgb_augmented = self.rgb_augment(image=image)
                    augmented_image = rgb_augmented['image']
                    
                    # 기하학적 변환 (Scale 제외, 회전/반전만 적용)
                    geometric_augmented = self.geometric_augment(
                        image=augmented_image,
                        mask=mask,
                        mono_depth_npy=mono_depth_npy,
                        mono_depth_png=mono_depth_png,
                        metric_depth_npy=metric_depth_npy,
                        metric_depth_png=metric_depth_png
                    )
                    
                    self._save_augmented_data(
                        geometric_augmented['image'],
                        geometric_augmented['mask'],
                        geometric_augmented['mono_depth_npy'],
                        geometric_augmented['mono_depth_png'],
                        geometric_augmented['metric_depth_npy'],
                        geometric_augmented['metric_depth_png'],
                        split,
                        original_name,
                        aug_idx
                    )
                    
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue
        
        print(f"✅ Completed augmenting {split} split")
    
    def augment_all(self):
        """train 폴더의 데이터를 증강합니다."""
        print("🚀 Starting dataset augmentation...")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"📁 Masks directory: {self.masks_dir}")
        print(f"📁 Output dataset: {self.output_root}")
        print(f"🔢 Augmentation multiplier: {self.multiplier}x")
        
        # train 폴더만 증강 수행
        self.augment_split('train')
        
        print("\n🎉 Dataset augmentation completed!")
        self._print_summary()
    
    def _print_summary(self):
        """증강 결과 요약 출력"""
        print("\n📊 Augmentation Summary:")
        print("=" * 50)
        
        images_dir = self.output_root / 'train' / 'images'
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*.png')))
            print(f"train: {num_images:>6} images")
        else:
            print(f"train: No images found")
        
        print("=" * 50)


# ============================================================================
# 명령행 인터페이스
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment RGB/Depth dataset using albumentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the images directory")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Path to the masks directory")
    parser.add_argument("--mono_depth_dir", type=str, default=None,
                        help="Path to the mono depth directory")
    parser.add_argument("--metric_depth_dir", type=str, default=None,
                        help="Path to the metric depth directory")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to save the augmented dataset")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Augmentation multiplier")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"❌ Error: Images directory does not exist: {images_dir}")
        sys.exit(1)
        
    parent_dir = images_dir.parent
    
    if args.masks_dir is None: masks_dir = parent_dir / "masks"
    else: masks_dir = Path(args.masks_dir)
        
    if args.mono_depth_dir is None: mono_depth_dir = parent_dir / "mono_depth"
    else: mono_depth_dir = Path(args.mono_depth_dir)
        
    if args.metric_depth_dir is None: metric_depth_dir = parent_dir / "metric_depth"
    else: metric_depth_dir = Path(args.metric_depth_dir)
    
    # 경로 검증
    if not masks_dir.exists():
        print(f"❌ Error: Masks directory does not exist: {masks_dir}")
        sys.exit(1)
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    augmentor = DatasetAugmentor(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mono_depth_dir=mono_depth_dir,
        metric_depth_dir=metric_depth_dir,
        output_root=output_root,
        multiplier=args.multiplier
    )
    
    augmentor.augment_all()


if __name__ == "__main__":
    main()