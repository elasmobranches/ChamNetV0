# ChamNet 데이터 전처리 파이프라인

참외 농장 자율주행 로봇을 위한 멀티태스크 러닝(MTL) 데이터셋 전처리 도구 모음입니다.

## 📋 프로젝트 개요

이 레포지토리는 RGB-D 데이터를 기반으로 세그멘테이션과 깊이 추정을 동시에 학습하기 위한 데이터 전처리 파이프라인을 제공합니다.

**주요 작업 흐름:**
1. COCO 형식 어노테이션 → 세그멘테이션 마스크 변환
2. 복잡한 파일명 단순화
3. 데이터 증강 (RGB, Mask, Depth 동기화)

## 🎯 대상 클래스

| 클래스 ID | 클래스명 | 색상 | 설명 |
|-----------|----------|------|------|
| 0 | background | Black | 배경 |
| 1 | chamoe | Yellow | 참외 |
| 2 | heatpipe | Red | 난방 파이프 |
| 3 | path | Green | 이동 경로 |
| 4 | pillar | Blue | 기둥 |
| 5 | topdownfarm | Magenta | 탑다운 농장 |
| 6 | unknown | Gray | 미분류 객체 |
| 7 | duct | - | 덕트 |

## 🛠️ 도구 구성

### 1. [coco_to_masks.py](coco_to_masks.py) - COCO 어노테이션 변환

COCO JSON 형식의 어노테이션을 세그멘테이션 마스크로 변환합니다.

**주요 기능:**
- 폴리곤/RLE 세그멘테이션 지원
- 클래스별 우선순위 처리 (겹침 해결)
- Grayscale 마스크 + Color 시각화 마스크 생성

**사용 예시:**
```bash
python coco_to_masks.py \
    --coco-json ./dataset/train/_annotations.coco.json \
    --out-dir ./dataset/train/masks_gray \
    --color-out-dir ./dataset/train/masks_color \
    --priority "3,5,2,1,4,6,7"
```

**우선순위 설정:**
- 기본값: `3,5,2,1,4,6,7` (path > topdownfarm > heatpipe > chamoe > pillar > unknown > duct)
- 나중에 오는 클래스가 먼저 오는 클래스를 덮어씁니다

### 2. [rename_files.py](rename_files.py) - 파일명 단순화

Roboflow 등에서 생성된 복잡한 파일명을 단순하게 변환합니다.

**변환 예시:**
```
이전: 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc.jpg
이후: 20250526_rfv4_frame_000298_00m_09s.jpg

이전: 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_mask.png
이후: 20250526_rfv4_frame_000298_00m_09s_mask.png
```

**사용 예시:**
```bash
# 미리보기 (실제 변경 안 함)
python rename_files.py --dataset_root ./dataset --dry_run

# 전체 파일명 변경
python rename_files.py --dataset_root ./dataset

# 특정 타입만 변경
python rename_files.py --dataset_root ./dataset --types image mask
```

**옵션:**
- `--dataset_root`: 데이터셋 루트 디렉토리
- `--types`: 변경할 파일 타입 (`image`, `mask`, `depth`, `mask_gray`, `mask_color`)
- `--splits`: 처리할 split (`train`, `val`, `test`)
- `--dry_run`: 미리보기 모드

### 3. [augmentation/](augmentation/) - 데이터 증강

RGB, Mask, Depth를 동기화하여 증강하는 도구입니다.

**특징:**
- RGB/Depth에 서로 다른 증강 기법 적용
- 기하학적 변환은 모든 모달리티에 동일 적용
- Hue 변환 제외로 참외 색상 보존
- 사용자 지정 배수 증강

**사용 예시:**
```bash
python augmentation/augmentation.py \
    --images_dir ./dataset/train/images \
    --output_root ./dataset_augmented \
    --multiplier 5
```

자세한 내용은 [augmentation/README_augmentation.md](augmentation/README_augmentation.md)를 참고하세요.

## 📁 데이터셋 구조

### 표준 디렉토리 구조
```
dataset_root/
├── train/
│   ├── images/                  # RGB 이미지
│   ├── masks/                   # 세그멘테이션 마스크 (학습용)
│   ├── masks_color/             # 컬러 시각화 마스크
│   ├── mono_depth/              # Monocular Depth (Depth Anything 등)
│   │   ├── depth_npy/           # numpy 형식 (.npy)
│   │   └── depth_visualization/ # 시각화 PNG
│   └── metric_depth/            # Metric Depth (Zed 2i 등)
│       ├── depth_npy/
│       └── depth_visualization/
├── val/
│   └── (train과 동일 구조)
└── test/
    └── (train과 동일 구조)
```

## 🔄 전처리 워크플로우

### 1단계: COCO 어노테이션 변환
```bash
# train/val/test 각각에 대해 실행
python coco_to_masks.py \
    --coco-json ./dataset/train/_annotations.coco.json \
    --out-dir ./dataset/train/masks \
    --color-out-dir ./dataset/train/masks_color
```

### 2단계: 파일명 단순화
```bash
# 미리보기로 먼저 확인
python rename_files.py --dataset_root ./dataset --dry_run

# 실제 변경
python rename_files.py --dataset_root ./dataset
```

### 3단계: 데이터 증강 (train만)
```bash
python augmentation/augmentation.py \
    --images_dir ./dataset/train/images \
    --masks_dir ./dataset/train/masks \
    --mono_depth_dir ./dataset/train/mono_depth \
    --metric_depth_dir ./dataset/train/metric_depth \
    --output_root ./dataset_augmented \
    --multiplier 5
```

## 📦 요구사항

```bash
pip install numpy opencv-python pillow albumentations tqdm
# COCO RLE 지원이 필요한 경우:
pip install pycocotools
```

## ⚙️ 환경

- **개발 환경**: RTX 4090 (GPU 성능에 크게 의존하지 않음)
- **OS**: Windows/Linux 모두 지원
- **Python**: 3.8+

## 📝 주의사항

1. **Depth 데이터 준비**: 증강 전에 Depth Anything 등으로 pseudo depth 생성 필요
2. **저장 공간**: 5배 증강 시 디스크 사용량 약 5배 증가
3. **파일 매칭**: RGB, Mask, Depth 파일의 베이스 이름이 일치해야 함
   - 예: `image001.jpg`, `image001_mask.png`, `image001_depth.npy`

## 🚀 빠른 시작

```bash
# 1. 저장소 클론
git clone <repository-url>
cd ChamNetV0-

# 2. 의존성 설치
pip install numpy opencv-python pillow albumentations tqdm pycocotools

# 3. COCO 어노테이션 변환
python coco_to_masks.py --coco-json ./dataset/train/_annotations.coco.json

# 4. 파일명 단순화
python rename_files.py --dataset_root ./dataset

# 5. 데이터 증강
python augmentation/augmentation.py \
    --images_dir ./dataset/train/images \
    --output_root ./dataset_augmented \
    --multiplier 5
```
