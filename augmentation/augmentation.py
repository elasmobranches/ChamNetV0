# ============================================================================
# 데이터셋 증강 스크립트 (Albumentations 기반)
# ============================================================================
# 이 스크립트는 train_esanet_mtl.py의 albumentations 증강 기법을 사용하여
# RGB/Depth 데이터셋을 증강하고 새로운 폴더에 저장합니다.
# 
# 주요 기능:
# - RGB와 Depth에 서로 다른 증강 기법 적용
# - 기하학적 변환은 RGB/Depth/Mask에 동일하게 적용
# - 색상 증강은 RGB에만 적용
# - 사용자가 지정한 배수만큼 데이터 증강

import argparse
import os
import sys
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
    coco_to_masks.py의 labels_to_color_image 함수를 참고하여 구현.
    
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
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.3
        ),
        
        # 가우시안 노이즈 추가
        A.GaussNoise(
            var_limit=(10.0, 50.0), 
            p=0.2
        ),
        
        # 감마 보정
        A.RandomGamma(
            gamma_limit=(80, 120), 
            p=0.2
        ),
        
        # HSV 색상 공간 변환
        A.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=30, 
            val_shift_limit=20, 
            p=0.2
        ),
        
        # 추가 색상 지터
        A.ColorJitter(
            brightness=0.1, 
            contrast=0.1, 
            saturation=0.1, 
            hue=0.05, 
            p=0.2
        ),
    ])


def get_geometric_augmentations():
    """
    기하학적 변환 증강 기법을 반환합니다.
    RGB, Depth, Mask에 동일하게 적용됩니다.
    
    Returns:
        A.Compose: 기하학적 변환용 albumentations 컴포즈
    """
    return A.Compose([
        # 좌우 반전
        A.HorizontalFlip(p=0.5),
        
        # 회전 (최대 15도)
        A.Rotate(limit=15, p=0.3),
        
        # 스케일링 (10% 범위)
        A.RandomScale(scale_limit=0.1, p=0.2),
        
        # 엘라스틱 변환 (자연스러운 왜곡) - 강도 조정
        A.ElasticTransform(
            alpha=25,  # 50 → 25로 강도 감소
            sigma=50, 
            p=0.05  # 0.1 → 0.05로 확률 감소
        ),
        
        # Cutout (부분 제거)
        A.CoarseDropout(
            max_holes=8, 
            max_height=32, 
            max_width=32, 
            p=0.1
        ),
    ], additional_targets={'depth': 'image', 'mask': 'mask'})


# ============================================================================
# 데이터셋 증강 클래스
# ============================================================================
class DatasetAugmentor:
    """
    RGB/Depth 데이터셋을 증강하는 클래스입니다.
    
    주요 기능:
    - RGB와 Depth에 서로 다른 증강 기법 적용
    - 기하학적 변환은 모든 데이터에 동일하게 적용
    - 색상 증강은 RGB에만 적용
    - 증강된 데이터를 새로운 폴더에 저장
    """
    
    def __init__(self, images_dir: Path, masks_dir: Path, depth_dir: Path, 
                 output_root: Path, multiplier: int):
        """
        데이터셋 증강기 초기화
        
        Args:
            images_dir: 이미지 디렉토리 경로
            masks_dir: 마스크 디렉토리 경로
            depth_dir: 깊이 디렉토리 경로
            output_root: 증강된 데이터셋 저장 경로
            multiplier: 증강 배수 (원본 데이터의 몇 배로 증강할지)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.depth_dir = Path(depth_dir)
        self.output_root = Path(output_root)
        self.multiplier = multiplier
        
        # 증강 기법 초기화
        self.rgb_augment = get_rgb_augmentations()
        self.geometric_augment = get_geometric_augmentations()
        
        # 출력 디렉토리 생성
        self._create_output_directories()
        
    def _create_output_directories(self):
        """출력 디렉토리 구조를 생성합니다."""
        # train 폴더에 대해 images, masks, masks_color, depth 폴더 생성
        for data_type in ['images', 'masks', 'masks_color', 'depth']:
            output_dir = self.output_root / 'train' / data_type
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {output_dir}")
    
    def _get_image_files(self, split: str) -> List[Path]:
        """
        특정 split의 이미지 파일 목록을 반환합니다.
        
        Args:
            split: 'train', 'val', 'test' 중 하나
            
        Returns:
            List[Path]: 이미지 파일 경로 리스트
        """
        # 사용자가 지정한 이미지 디렉토리 사용
        images_dir = self.images_dir
        if not images_dir.exists():
            print(f"⚠️ Warning: {images_dir} does not exist")
            return []
        
        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def _load_image_data(self, image_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        이미지와 관련된 마스크, 깊이 데이터를 로드합니다.
        
        Args:
            image_path: 이미지 파일 경로
            split: 데이터셋 split
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (image, mask, depth) numpy 배열
        """
        # 파일명에서 stem 추출
        img_stem = image_path.stem
        
        # 마스크 파일 경로 - PNG 확장자 사용
        mask_path = self.masks_dir / f"{img_stem}_mask.png"
        
        # 깊이 파일 경로 - PNG 확장자 사용
        depth_path = self.depth_dir / f"{img_stem}_depth.png"
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 로드 - 그레이스케일로 로드
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
        else:
            print(f"⚠️ Warning: Mask file not found: {mask_path}")
            # 빈 마스크 생성
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 깊이 로드 - 원본 비트 심도 유지
        if depth_path.exists():
            # 원본 데이터 타입을 유지하기 위해 IMREAD_UNCHANGED 사용
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to load depth: {depth_path}")
            
            # 깊이가 3채널인 경우 첫 번째 채널만 사용
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            
            # 데이터 타입 정규화 (16비트 → 8비트 변환 시 정보 손실 방지)
            if depth.dtype == np.uint16:
                # 16비트 데이터를 8비트로 정규화 (0-65535 → 0-255)
                depth = (depth / 256).astype(np.uint8)
            elif depth.dtype == np.uint8:
                # 이미 8비트인 경우 그대로 사용
                pass
            else:
                # 기타 타입은 uint8로 변환
                depth = depth.astype(np.uint8)
        else:
            print(f"⚠️ Warning: Depth file not found: {depth_path}")
            # 빈 깊이 맵 생성
            depth = np.zeros(image.shape[:2], dtype=np.uint8)
        
        return image, mask, depth
    
    def _save_augmented_data(self, image: np.ndarray, mask: np.ndarray, depth: np.ndarray, 
                           split: str, original_name: str, aug_idx: int):
        """
        증강된 데이터를 저장합니다.
        
        Args:
            image: 증강된 RGB 이미지
            mask: 증강된 마스크
            depth: 증강된 깊이 맵
            split: 데이터셋 split
            original_name: 원본 파일명 (확장자 제외)
            aug_idx: 증강 인덱스
        """
        # 파일명 생성 - 모든 파일을 PNG로 저장
        base_name = f"{original_name}_aug_{aug_idx:03d}"
        
        # RGB 이미지 저장 - PNG로 저장
        image_path = self.output_root / split / 'images' / f"{base_name}.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)
        
        # 그레이스케일 마스크 저장 - PNG로 저장
        mask_path = self.output_root / split / 'masks' / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # 컬러 마스크 생성 및 저장
        color_mask = labels_to_color_image(mask)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        color_mask_path = self.output_root / split / 'masks_color' / f"{base_name}_mask.png"
        cv2.imwrite(str(color_mask_path), color_mask_bgr)
        
        # 깊이 맵 저장 - PNG로 저장, 데이터 타입 고려
        depth_path = self.output_root / split / 'depth' / f"{base_name}_depth.png"
        
        # 깊이 데이터 타입에 따른 저장 최적화
        if depth.dtype == np.uint16:
            # 16비트 데이터는 그대로 저장 (더 높은 정밀도)
            cv2.imwrite(str(depth_path), depth)
        else:
            # 8비트 데이터는 기본 저장
            cv2.imwrite(str(depth_path), depth)
    
    def augment_split(self, split: str):
        """
        특정 split의 데이터를 증강합니다.
        
        Args:
            split: 'train', 'val', 'test' 중 하나
        """
        print(f"\n🔄 Augmenting {split} split...")
        
        # 이미지 파일 목록 가져오기
        image_files = self._get_image_files(split)
        if not image_files:
            print(f"⚠️ No images found in {split} split")
            return
        
        print(f"📊 Found {len(image_files)} images in {split} split")
        
        # 각 이미지에 대해 증강 수행
        for img_idx, image_path in enumerate(tqdm(image_files, desc=f"Augmenting {split}")):
            try:
                # 원본 데이터 로드
                image, mask, depth = self._load_image_data(image_path, split)
                original_name = image_path.stem
                
                # 원본 데이터 복사 (aug_000) - PNG로 저장
                self._save_augmented_data(image, mask, depth, split, original_name, 0)
                
                # 증강 수행
                for aug_idx in range(1, self.multiplier):
                    # RGB 색상 증강 적용
                    rgb_augmented = self.rgb_augment(image=image)
                    augmented_image = rgb_augmented['image']
                    
                    # 기하학적 변환 적용 (RGB, Depth, Mask 모두 동일하게)
                    geometric_augmented = self.geometric_augment(
                        image=augmented_image,
                        mask=mask,
                        depth=depth
                    )
                    
                    # 증강된 데이터 저장 - PNG로 저장
                    self._save_augmented_data(
                        geometric_augmented['image'],
                        geometric_augmented['mask'],
                        geometric_augmented['depth'],
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
        print(f"📁 Depth directory: {self.depth_dir}")
        print(f"📁 Output dataset: {self.output_root}")
        print(f"🔢 Augmentation multiplier: {self.multiplier}x")
        
        # train 폴더만 증강 수행
        self.augment_split('train')
        
        print("\n🎉 Dataset augmentation completed!")
        self._print_summary()
    
    def _print_summary(self):
        """증강 결과 요약을 출력합니다."""
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
    """
    명령행 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인수
    """
    parser = argparse.ArgumentParser(
        description="Augment RGB/Depth dataset using albumentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to the images directory"
    )
    
    parser.add_argument(
        "--masks_dir", 
        type=str,
        required=True,
        help="Path to the masks directory"
    )
    
    parser.add_argument(
        "--depth_dir",
        type=str, 
        required=True,
        help="Path to the depth directory"
    )
    
    parser.add_argument(
        "--output_root", 
        type=str, 
        required=True,
        help="Path to save the augmented dataset"
    )
    
    parser.add_argument(
        "--multiplier", 
        type=int, 
        default=5,
        help="Augmentation multiplier (how many times to augment each image)"
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 경로 검증
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    depth_dir = Path(args.depth_dir)
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory does not exist: {images_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"❌ Error: Masks directory does not exist: {masks_dir}")
        sys.exit(1)
    if not depth_dir.exists():
        print(f"❌ Error: Depth directory does not exist: {depth_dir}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 증강기 초기화
    augmentor = DatasetAugmentor(
        images_dir=images_dir,
        masks_dir=masks_dir,
        depth_dir=depth_dir,
        output_root=output_root,
        multiplier=args.multiplier
    )
    
    # 증강 수행
    augmentor.augment_all()
    
    print(f"\n🎯 Augmentation completed!")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Masks directory: {masks_dir}")
    print(f"📁 Depth directory: {depth_dir}")
    print(f"📁 Augmented dataset: {output_root}")
    print(f"🔢 Multiplier: {args.multiplier}x")


if __name__ == "__main__":
    main()
