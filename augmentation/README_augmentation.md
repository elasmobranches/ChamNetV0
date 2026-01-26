# 데이터셋 증강 도구 (Dataset Augmentation Tool)

이 도구는 RGB/Depth 데이터셋을 증강하는 스크립트입니다. MTL(Multi-Task Learning) 학습을 위해 RGB, Mask, Mono Depth, Metric Depth를 동기화하여 증강합니다.

RTX 4090 환경에서 작업하였으나 환경에 크게 의존하지 않습니다. 
사용하기 전에 depth anything을 이용하여 pseudo depth 전환이 필요합니다.
증강은 train 폴더만 진행됩니다.

## 🎯 주요 기능

- **RGB와 Depth에 서로 다른 증강 기법 적용**
- **기하학적 변환은 RGB/Depth/Mask에 동일하게 적용**
- **색상 증강은 RGB에만 적용** (Depth 값 보존)
- **Hue(색조) 변환 제외**로 참외의 노란색 등 핵심 색상 보존
- **사용자가 지정한 배수만큼 데이터 증강**
- **자동 디렉토리 구조 생성**

## 📁 데이터셋 구조

### 입력 데이터셋 구조
```
dataset_root/
├── images/              # RGB 이미지
├── masks/               # 세그멘테이션 마스크 (*_mask.png)
├── mono_depth/          # Monocular Depth (Depth Anything 등)
│   ├── depth_npy/       # numpy 형식 (*_depth.npy)
│   └── depth_visualization/  # 시각화용 PNG (*_depth.png)
└── metric_depth/        # Metric Depth (Zed 2i 등)
    ├── depth_npy/
    └── depth_visualization/
```

### 출력 데이터셋 구조
```
output_root/
└── train/
    ├── images/          # 증강된 RGB 이미지
    ├── masks/           # 증강된 마스크
    ├── masks_color/     # 컬러 시각화 마스크
    ├── mono_depth/
    │   ├── depth_npy/
    │   └── depth_visualization/
    └── metric_depth/
        ├── depth_npy/
        └── depth_visualization/
```

## 🔧 증강 기법

### RGB 전용 색상 증강
| 기법 | 설명 | 확률 |
|------|------|------|
| **RandomBrightnessContrast** | 밝기/대비 ±10% 조정 | p=0.3 |
| **ImageCompression** | JPEG 압축 아티팩트 (품질 80-100) | p=0.3 |
| **MotionBlur** | 로봇 이동 시 발생하는 흐림 효과 | p=0.2 |
| **RandomGamma** | 감마 보정 (90-110) | p=0.2 |
| **HueSaturationValue** | 채도/명도 변환 (Hue 고정) | p=0.2 |
| **ColorJitter** | 추가 색상 지터 (Hue 고정) | p=0.2 |

> **참고**: Hue(색조) 변환은 비활성화되어 참외의 노란색 등 핵심 색상이 보존됩니다.

### 기하학적 변환 (RGB/Depth/Mask 공통)
| 기법 | 설명 | 확률 |
|------|------|------|
| **HorizontalFlip** | 좌우 반전 | p=0.5 |
| **Rotate** | 회전 (최대 ±15도, border=0) | p=0.3 |

> **제거된 증강**: RandomScale, ElasticTransform, CoarseDropout은 Depth estimation에 부적합하여 제거되었습니다.

## 🚀 사용법

### 기본 사용법
```bash
python augmentation.py \
    --images_dir ./dataset/images \
    --output_root ./dataset_augmented \
    --multiplier 5
```

### 전체 옵션 지정
```bash
python augmentation.py \
    --images_dir ./dataset/images \
    --masks_dir ./dataset/masks \
    --mono_depth_dir ./dataset/mono_depth \
    --metric_depth_dir ./dataset/metric_depth \
    --output_root ./dataset_augmented \
    --multiplier 5 \
    --seed 42
```

## 📋 명령행 옵션

| 옵션 | 설명 | 기본값 | 필수 |
|------|------|--------|------|
| `--images_dir` | RGB 이미지 디렉토리 경로 | - | ✅ |
| `--masks_dir` | 마스크 디렉토리 경로 | images 상위/masks | ❌ |
| `--mono_depth_dir` | Mono depth 디렉토리 경로 | images 상위/mono_depth | ❌ |
| `--metric_depth_dir` | Metric depth 디렉토리 경로 | images 상위/metric_depth | ❌ |
| `--output_root` | 증강된 데이터셋 저장 경로 | - | ✅ |
| `--multiplier` | 증강 배수 | 5 | ❌ |
| `--seed` | 랜덤 시드 (재현성) | 42 | ❌ |

## 📊 출력 파일명 규칙

원본 파일: `image001.jpg`
- `image001_aug_000.png` — 원본 (증강 없음)
- `image001_aug_001.png` — 1번째 증강
- `image001_aug_002.png` — 2번째 증강
- ...
- `image001_aug_004.png` — 4번째 증강 (5배 증강 시)

## 🔍 예제

### 5배 증강
```bash
# 원본: 100개 이미지 → 증강 후: 500개 이미지
python augmentation.py \
    --images_dir ../dataset/images \
    --output_root ../dataset_5x \
    --multiplier 5
```

## ⚠️ 주의사항

1. **저장 공간**: 5배 증강 시 디스크 사용량이 약 5배 증가합니다.
2. **파일 형식**: RGB/Mask/Depth 시각화는 PNG, Depth 데이터는 NPY로 저장됩니다.
3. **누락 파일 처리**: Mask나 Depth 파일이 없으면 경고 후 빈 배열로 대체됩니다.
4. **Rotation border**: 회전 시 생기는 빈 영역은 0으로 채워집니다 (Depth에서 invalid 의미).

## 🛠️ 요구사항
```bash
pip install albumentations opencv-python pillow tqdm numpy
```

## 🎯 설계 근거

- **Hue 변환 제외**: 참외(노란색), 난방파이프(빨간색) 등 색상이 클래스의 핵심 특징이므로 색조 변환을 비활성화
- **MotionBlur 추가**: 로봇이 이동하면서 촬영하는 실제 환경 반영
- **ElasticTransform/CoarseDropout 제거**: Depth map의 연속성을 인위적으로 왜곡하여 학습에 악영향
- **RandomScale 제거**: 이미지 크기 변화로 배치 구성 실패 방지