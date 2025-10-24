# 데이터셋 증강 도구 (Dataset Augmentation Tool)

이 도구는 `train_esanet_mtl.py`에서 사용하는 albumentations 증강 기법을 기반으로 RGB/Depth 데이터셋을 증강하는 스크립트입니다.

## 🎯 주요 기능

- **RGB와 Depth에 서로 다른 증강 기법 적용**
- **기하학적 변환은 RGB/Depth/Mask에 동일하게 적용**
- **색상 증강은 RGB에만 적용**
- **사용자가 지정한 배수만큼 데이터 증강**
- **자동 디렉토리 구조 생성**

## 📁 데이터셋 구조

### 입력 데이터셋 구조
```
dataset_root/
├── train/
│   ├── images/     # RGB 이미지
│   ├── masks/      # 세그멘테이션 마스크
│   └── depth/      # 깊이 맵
├── val/
│   ├── images/
│   ├── masks/
│   └── depth/
└── test/
    ├── images/
    ├── masks/
    └── depth/
```

### 출력 데이터셋 구조
```
output_root/
├── train/
│   ├── images/     # 증강된 RGB 이미지
│   ├── masks/      # 증강된 마스크
│   └── depth/      # 증강된 깊이 맵
├── val/
│   ├── images/
│   ├── masks/
│   └── depth/
└── test/
    ├── images/
    ├── masks/
    └── depth/
```

## 🔧 증강 기법

### RGB 전용 색상 증강
- **RandomBrightnessContrast**: 밝기/대비 조정 (p=0.3)
- **GaussNoise**: 가우시안 노이즈 추가 (p=0.2)
- **RandomGamma**: 감마 보정 (p=0.2)
- **HueSaturationValue**: HSV 색상 변환 (p=0.2)
- **ColorJitter**: 추가 색상 지터 (p=0.2)

### 기하학적 변환 (RGB/Depth/Mask 공통)
- **HorizontalFlip**: 좌우 반전 (p=0.5)
- **Rotate**: 회전 (최대 15도, p=0.3)
- **RandomScale**: 스케일링 (10% 범위, p=0.2)
- **ElasticTransform**: 엘라스틱 변환 (p=0.1)
- **CoarseDropout**: Cutout 효과 (p=0.1)

## 🚀 사용법

### 기본 사용법
```bash
python augmentation.py \
    --dataset_root ../dataset \
    --output_root ../dataset_augmented \
    --multiplier 5
```

### 특정 split만 증강
```bash
python augmentation.py \
    --dataset_root ../dataset \
    --output_root ../dataset_augmented \
    --multiplier 3 \
    --splits train val
```

### 예제 스크립트 실행
```bash
python run_augmentation.py
```

## 📋 명령행 옵션

| 옵션 | 설명 | 기본값 | 필수 |
|------|------|--------|------|
| `--dataset_root` | 원본 데이터셋 루트 경로 | - | ✅ |
| `--output_root` | 증강된 데이터셋 저장 경로 | - | ✅ |
| `--multiplier` | 증강 배수 (원본의 몇 배로 증강할지) | 5 | ❌ |
| `--splits` | 증강할 split 선택 | train val test | ❌ |

## 📊 출력 파일명 규칙

원본 파일: `image001.jpg`
- `image001_aug_000.png` (원본 복사)
- `image001_aug_001.png` (1번째 증강)
- `image001_aug_002.png` (2번째 증강)
- ...

## 🔍 예제

### 5배 증강 예제
```bash
# 원본: 100개 이미지
# 증강 후: 500개 이미지 (100 × 5)
python augmentation.py \
    --dataset_root ../dataset \
    --output_root ../dataset_5x \
    --multiplier 5
```

### train split만 3배 증강
```bash
python augmentation.py \
    --dataset_root ../dataset \
    --output_root ../dataset_train_3x \
    --multiplier 3 \
    --splits train
```

## ⚠️ 주의사항

1. **메모리 사용량**: 증강 배수가 클수록 더 많은 메모리가 필요합니다.
2. **저장 공간**: 증강된 데이터셋은 원본보다 훨씬 클 수 있습니다.
3. **파일 형식**: 출력은 모두 PNG 형식으로 저장됩니다.
4. **마스크/깊이 파일**: 해당 파일이 없으면 빈 이미지로 생성됩니다.

## 🛠️ 요구사항

```bash
pip install albumentations opencv-python pillow tqdm numpy
```

## 📈 성능 최적화

- **병렬 처리**: 대용량 데이터셋의 경우 멀티프로세싱 고려
- **메모리 관리**: 배치 단위로 처리하여 메모리 사용량 제어
- **디스크 I/O**: SSD 사용 권장

## 🎯 사용 사례

1. **데이터 부족 해결**: 작은 데이터셋을 확장
2. **모델 강건성 향상**: 다양한 변형으로 일반화 능력 개선
3. **클래스 불균형 해결**: 특정 클래스의 데이터 증강
4. **실험용 데이터 생성**: 다양한 증강 설정으로 실험
