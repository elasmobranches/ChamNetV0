# 🚀 ESANet 멀티태스크 학습 프레임워크

---

## 🎯 **개요**

이 저장소는 RGB-D 데이터를 활용한 **멀티태스크 학습(Multi-Task Learning, MTL)**을 위한 완전한 프레임워크를 제공합니다. ESANet 아키텍처를 기반으로 **의미 분할(Semantic Segmentation)**과 **깊이 추정(Depth Estimation)**을 동시에 수행할 수 있습니다. **불확실성 가중치(Uncertainty Weighting)**와 **동적 가중치 평균(Dynamic Weight Average, DWA)** 등 최신 기법을 통해 최적의 태스크 균형을 달성합니다.

### ✨ **주요 특징**

- 🔥 **멀티태스크 학습**: 의미 분할과 깊이 추정 동시 수행
- ⚡ **고급 가중치 조정**: 불확실성 기반 및 DWA 태스크 균형화
- 🎨 **데이터 증강**: RGB-D 데이터를 위한 종합적인 증강 파이프라인
- 📊 **성능 분석**: FLOPs 측정 및 추론 도구
- 🚀 **프로덕션 준비**: 완전한 학습, 검증, 추론 파이프라인
- 📈 **최신 기술**: PyTorch Lightning과 torchmetrics 통합

---

## 📁 **저장소 구조**

```
esanet-mtl/
├── 📂 models/                    # 핵심 학습 스크립트
│   ├── train_esanet_mtl_uncertain.py    # 불확실성 가중치 MTL
│   ├── train_esanet_mtl_dwa.py          # 동적 가중치 평균 MTL
│   ├── train_esanet_seg_only.py         # 의미 분할 전용 학습
│   └── train_esanet_depth_only.py       # 깊이 추정 전용 학습
├── 📂 augmentation/              # 데이터 증강 도구
│   ├── augmentation.py                  # RGB-D 증강 파이프라인
│   └── README_augmentation.md          # 증강 문서
├── 📂 flops_models/              # 성능 분석
│   ├── measure_flops_mtl.py             # MTL 모델 FLOPs 측정
│   ├── measure_flops_seg_only.py        # 의미 분할 모델 분석
│   └── measure_flops_depth_only.py      # 깊이 모델 분석
├── 📂 infer_models/              # 추론 스크립트(시각화는 삭제. FPS 확인용)
│   ├── infer_single_mtl.py              # 단일 이미지 MTL 추론
│   ├── infer_single_seg.py              # 단일 이미지 의미 분할
│   ├── infer_single_depth.py            # 단일 이미지 깊이 추정
│   └── infer_sequential_seg_depth.py    # 순차적 추론 파이프라인
└── 📄 README.md                  # 이 파일
```

---

## 🚀 **빠른 시작**

### **1. 설치**

```bash
# 저장소 클론
git clone https://github.com/elasmobranches/ChamNetV0.git
cd ChamNetV0

# 의존성 설치
pip install -r requirements.txt
```

### **2. 데이터 준비**

```bash
# 다음 구조로 데이터셋 준비:
dataset/
├── train/
│   ├── images/     # RGB 이미지 (.jpg, .png)
│   ├── masks/      # 의미 분할 마스크 (.png)
│   └── depth/      # 깊이 맵 (.png)
├── val/
│   ├── images/
│   ├── masks/
│   └── depth/
└── test/
    ├── images/
    ├── masks/
    └── depth/
```

### **3. 데이터 증강 (선택사항)**

```bash
#  ex) 데이터셋을 5배 증강
python augmentation/augmentation.py \
    --dataset_root ./dataset \
    --output_root ./dataset_augmented \
    --multiplier 5
```

### **4. 학습**

```bash
# 불확실성 가중치를 사용한 멀티태스크 학습
python models/train_esanet_mtl_uncertain.py --config configs/uncertain.yaml

# DWA를 사용한 멀티태스크 학습
python models/train_esanet_mtl_dwa.py --config configs/dwa.yaml

# 단일 태스크 학습
python models/train_esanet_seg_only.py --config configs/seg.yaml
python models/train_esanet_depth_only.py --config configs/depth.yaml
```

### **5. 추론**

```bash
# 단일 이미지 추론
python infer_models/infer_single_mtl.py \
    --ckpt ./checkpoints/best_model.ckpt \
    --rgb ./test_image.jpg \
    --depth ./test_depth.png \
    --height 512 --width 512
```

---

## 🧠 **아키텍처**

### **모델 구조 및 Import**

이 프레임워크의 핵심 모델인 `ESANet`은 다음과 같이 import하여 사용할 수 있습니다:

```python
from src.models.model import ESANet
```

`model.py` 파일은 다음과 같은 의존성을 가집니다:
- `src.models.resnet`: ResNet18, ResNet34, ResNet50 백본 네트워크
- `src.models.rgb_depth_fusion`: SqueezeAndExciteFusionAdd 융합 모듈
- `src.models.context_modules`: 컨텍스트 모듈 (PPM 등)
- `src.models.model_utils`: 유틸리티 함수들 (ConvBNAct, Swish, Hswish)

### **ESANet 멀티태스크 아키텍처**

<div align="center">
  <img src="https://via.placeholder.com/800x400/2E3440/ECEFF4?text=ESANet+멀티태스크+아키텍처" alt="ESANet Architecture" width="80%"/>
</div>

이 프레임워크는 **ESANet**을 백본으로 사용하여 RGB와 깊이 정보를 효율적으로 처리합니다:

- **RGB 인코더**: 색상 정보 처리
- **깊이 인코더**: 깊이 정보 처리  
- **공유 디코더**: 다중 모달 특징 융합
- **태스크 헤드**: 의미 분할과 깊이 추정을 위한 별도 헤드

### **멀티태스크 학습 전략**

#### **1. 불확실성 가중치** 
[Kendall et al. (CVPR 2018)](https://arxiv.org/abs/1705.07115) 기반:

```
L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log(σ₁) + log(σ₂)
```

#### **2. 동적 가중치 평균 (DWA)**
[Liu et al. (CVPR 2019)](https://arxiv.org/abs/1803.10704) 기반:

- 상대적 손실 감소율을 기반으로 태스크 가중치 자동 조정
- 학습 중 태스크 불균형 방지

#### 실험을 통해 동적 가중치 사용 결정
---

## 📊 **성능 분석**

### **FLOPs 측정**

```bash
# 모델 복잡도 측정
python flops_models/measure_flops_mtl.py \
    --height 512 --width 512 \
    --batch_size 1

# 다양한 아키텍처 비교
python flops_models/measure_flops_seg_only.py
python flops_models/measure_flops_depth_only.py
```

### **예상 성능**

| 모델 | 파라미터 (M) | mIoU | AbsRel |
|------|-------------|------|--------|
| MTL (DWA) | 31.3 | 0.79 | 0.070 |
| Seg Only | 16.3 | 0.77 | - |
| Depth Only | 16.3 | - | 0.191 |

---

## 🎨 **데이터 증강**

RGB-D 데이터를 위해 특별히 설계된 종합적인 증강 파이프라인을 포함합니다:

### **RGB 증강**
- **색상 지터링**: 밝기, 대비, 채도, 색조 조정
- **노이즈 추가**: 가우시안 노이즈, 색상 지터
- **감마 보정**: 랜덤 감마 조정

### **기하적 증강** (RGB, 깊이, 마스크에 적용)
- **공간 변환**: 회전, 스케일링, 뒤집기
- **탄성 변형**: 비강체 변환
- **컷아웃**: 랜덤 영역 마스킹

#### depth image는 기하학적 증강만, rgb image는 RGB 증강 추가
```


## 🔧 **설정**

### **YAML 설정 예제**

```yaml
# configs/uncertain.yaml
model:
  height: 512
  width: 512
  num_classes: 7
  encoder_rgb: "resnet34"
  encoder_depth: "resnet34"
  use_pretrained: true

training:
  batch_size: 8
  epochs: 200
  lr: 1e-4
  use_uncertainty_weighting: false
  use_dwa_weighting: True
  early_stop_patience: 20

data:
  dataset_root: "./dataset"
  num_workers: 4
  pin_memory: true

system:
  precision: 16
  accelerator: "gpu"
  devices: 1
```