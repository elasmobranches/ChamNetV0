# ChamNet: 사족보행로봇의 온실 환경 주행을 위한 멀티태스크 러닝 기반 의미론적 분할 및 깊이 추정 연구

MMSegmentation과 MMDeploy 기반의 농업 환경에서 의미론적 분할(Semantic Segmentation)과 깊이 추정(Depth Estimation)을 동시에 수행하는 Multi-Task Learning 프레임워크입니다.

## 개요

이 저장소는 농업 이미지에서 의미론적 분할(7개 클래스)과 단안 깊이 추정을 공동으로 수행하는 MTL 모델의 학습, 평가, 배포를 위한 완전한 파이프라인을 제공합니다. SegFormer (MiT-B0)를 공유 백본으로 사용하며 태스크별 상호작용 헤드를 통해 특징을 공유합니다.

### 주요 기능

- **Multi-Task Learning**: 다양한 가중치 전략(Uncertainty, DWA, Manual)을 통한 분할 및 깊이 추정의 공동 최적화
- **유연한 아키텍처**: 태스크 간 특징 공유를 가능하게 하는 상호작용 기반 디코더 헤드
- **완전한 파이프라인**: 학습, 평가, 벤치마킹, FLOPs 계산의 자동화
- **배포 준비 완료**: 엣지 디바이스(Jetson Orin)를 위한 TensorRT 최적화를 통한 ONNX 내보내기
- **확장 가능**: MTL을 위한 커스텀 모듈을 포함한 MMSegmentation 생태계 기반

### 지원 태스크

| 태스크 | 출력 | 평가 지표 |
|------|------|---------|
| 의미론적 분할 | 7개 클래스 (background, chamoe, heatpipe, path, pillar, topdownfarm, unknown) | mIoU, Dice Loss |
| 깊이 추정 | 메트릭 깊이 (0-10m) | abs_rel, sq_rel, RMSE, δ1/δ2/δ3 |

---

## 프로젝트 구조

```
ChamNetV0-/
├── configs/                    # 모델 및 학습 설정 파일
│   ├── chamnet/               # ChamNet MTL 모델 설정
│   │   ├── chamnet_mtl_base_segformer_chamdatav3.py
│   │   └── chamnet_mtl_base_segformer_chamdatav4.py
│   └── _base_/                # 기본 설정 (데이터셋, 스케줄, 런타임)
│       ├── chamdata_mtl.py    # MTL 데이터셋 설정
│       ├── schedule_30m.py    # 학습 스케줄 (30K iterations)
│       └── default_runtime.py # 런타임 설정
│
├── mmseg/                      # 커스텀 MMSegmentation 모듈
│   ├── models/
│   │   ├── segmentors/        # MTL 모델 아키텍처
│   │   │   ├── mtl_encoder_decoder.py          # 기본 MTL 인코더-디코더
│   │   │   └── mtl_interaction_segmentor.py    # 상호작용 기반 MTL 모델
│   │   ├── decode_heads/      # 태스크별 디코더 헤드
│   │   │   ├── interaction_segformer_head.py   # 특징 내보내기 기능을 가진 분할 헤드
│   │   │   └── depth_head.py                   # 태스크 간 특징을 활용하는 깊이 헤드
│   │   └── losses/            # 커스텀 손실 함수
│   │       └── depth_losses.py                 # SILog, BerHu 손실 함수
│   ├── datasets/              # MTL 데이터셋 구현
│   │   └── mtl_dataset.py     # 분할 + 깊이를 위한 커스텀 데이터셋
│   ├── evaluation/            # 평가 메트릭
│   │   └── metrics/
│   │       ├── iou_metric.py  # 분할 mIoU
│   │       └── depth_metric.py # 깊이 메트릭 (abs_rel, RMSE 등)
│   ├── engine/                # 학습 엔진 컴포넌트
│   │   └── mtl_hooks.py       # MTL 전용 학습 훅
│   └── data_preprocessor.py   # MTL 데이터 전처리
│
├── hooks/                      # 커스텀 학습 훅
│   ├── custom_hooks.py        # 일반 커스텀 훅
│   ├── iter_logger_hook.py    # 반복(iteration)별 로깅
│   ├── mtl_iter_logger_hook.py # MTL 전용 로깅
│   └── depth_iter_logger_hook.py # 깊이 전용 로깅
│
├── mmdeploy/                   # ONNX 내보내기 및 배포
│   ├── mtl_deploy_cfg.py      # ONNX/TensorRT 배포 설정
│   ├── deploy_model_cfg.py    # 모델 배포 설정
│   ├── mmdeploy_test.py       # 배포 테스트
│   └── robot_main.py          # 로봇 추론 진입점
│
├── tools/                      # 학습 및 분석 유틸리티
│   ├── train.py               # 학습 스크립트
│   ├── test.py                # 평가 스크립트
│   ├── run_full_pipeline_mtl.py # 자동화된 전체 파이프라인 (학습→테스트→벤치마크)
│   └── analysis_tools/
│       ├── benchmark.py       # FPS 벤치마킹
│       └── get_flops_universal.py # FLOPs 계산
│
├── inference.py                # 시각화를 포함한 추론 스크립트
└── inference_result/           # 추론 결과 출력 디렉토리
```

---

## 설치

### 요구사항

- Python 3.8+
- CUDA 지원 PyTorch 1.12+
- MMSegmentation 1.x
- MMDeploy (ONNX 내보내기용)

### 설정

```bash
# 저장소 클론
git clone <repository-url>
cd ChamNetV0-

# 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mmengine mmcv mmsegmentation

# 배포용 (선택사항)
pip install mmdeploy mmdeploy-runtime
```

---

## 사용법

### 1. 학습

다양한 손실 가중치 전략으로 MTL 모델 학습:

```bash
# Uncertainty Weighting (권장)
python tools/run_full_pipeline_mtl.py \
    configs/chamnet/chamnet_mtl_base_segformer_chamdatav3.py \
    --weight-strategy uncertainty

# Dynamic Weight Average (DWA)
python tools/run_full_pipeline_mtl.py \
    configs/chamnet/chamnet_mtl_base_segformer_chamdatav3.py \
    --weight-strategy dwa

# Manual Weighting
python tools/run_full_pipeline_mtl.py \
    configs/chamnet/chamnet_mtl_base_segformer_chamdatav3.py \
    --weight-strategy manual --seg-weight 1.0 --depth-weight 5.0
```

파이프라인이 자동으로 수행하는 작업:
- 모델 학습 (30K iterations)
- 테스트 세트 검증
- FPS 벤치마킹
- FLOPs 계산
- 학습 곡선 시각화

### 2. 평가만 수행

```bash
python tools/test.py \
    configs/chamnet/chamnet_mtl_base_segformer_chamdatav3.py \
    work_dirs/chamnet_mtl_uncertainty/best_depth_abs_rel_iter_*.pth
```

### 3. 추론

시각화를 포함한 커스텀 이미지 추론:

```bash
python inference.py \
    work_dirs/chamnet_mtl_uncertainty/best_depth_abs_rel_iter_*.pth \
    --img-folder path/to/images \
    --mask-folder path/to/masks \
    --depth-folder path/to/depth \
    --output-dir ./visualizations
```

출력: 2×3 그리드 시각화 (GT 분할 | 입력 RGB | 예측 분할 / GT 깊이 | 입력 RGB | 예측 깊이)

### 4. ONNX 내보내기

배포를 위해 학습된 모델을 ONNX 형식으로 내보내기:

```bash
# ONNX로 내보내기 (FP32)
python mmdeploy/deploy_model_cfg.py

# TensorRT로 변환 (Jetson Orin용 FP16)
# TensorRT 설정은 mmdeploy/mtl_deploy_cfg.py 참조
```

---

## 모델 아키텍처

```
입력 RGB (512×512)
    ↓
┌──────────────────────┐
│  공유 백본           │  ← SegFormer MiT-B0
│  (MixVisionTransformer) │
└──────────────────────┘
    ↓ [32, 64, 160, 256]
    ├────────────────────┬────────────────────┐
    ↓                    ↓                    ↓
┌──────────────┐   ┌─────────────────┐   ┌──────────────┐
│ 분할 디코더  │   │  특징 내보내기  │   │ 깊이 디코더  │
│ (SegformerHead)│→│  (seg_feat)    │→│ (DepthHead)  │
└──────────────┘   └─────────────────┘   └──────────────┘
    ↓                                          ↓
분할 결과 (7 classes)                  깊이 맵 (0-10m)
```

**상호작용 메커니즘**: 분할 헤드가 중간 특징(`seg_feat`)을 내보내고, 이를 깊이 디코더로 전달하여 깊이 태스크가 의미론적 정보를 활용할 수 있도록 합니다.

---

## 손실 가중치 전략

| 전략 | 설명 | 사용 사례 |
|------|------|----------|
| **Uncertainty** | 태스크별 불확실성 매개변수 학습 (homoscedastic) | 최고 성능을 위해 권장 |
| **DWA** | 태스크 수렴 속도 기반 동적 가중치 평균 | 실험적, 다양한 태스크 난이도 처리 |
| **Manual** | 사용자 지정 고정 가중치 | 세밀한 제어, 도메인별 튜닝 |

**손실 함수**:
```
Total Loss = w_seg × (CE + Dice) + w_depth × (SILog + BerHu)
```

여기서 `w_seg`와 `w_depth`는 선택한 전략에 따라 결정됩니다.

---

## 데이터셋 형식

예상되는 디렉토리 구조:

```
dataset/
├── train/
│   ├── images/           # RGB 이미지 (.png)
│   ├── masks/            # 분할 마스크 (*_mask.png)
│   └── metric_depth/
│       └── depth_npy/    # 깊이 맵 (*_depth.npy)
├── valid/
│   └── ...
└── test/
    └── ...
```

- **Images**: RGB 이미지 (512×512 또는 자동 리사이즈)
- **Masks**: 단일 채널 PNG (7개 클래스를 위한 픽셀 값 0-6)
- **Depth**: 미터 단위 메트릭 깊이 (0-10m 범위)가 포함된 NumPy 배열 (.npy)

---

## 결과

ChamData 검증 세트에서의 성능:

| 모델 | 백본 | mIoU (%) | abs_rel | RMSE | FPS (GPU) | FLOPs (G) |
|------|------|----------|---------|------|-----------|-----------|
| ChamNet-MTL | SegFormer-B0 | TBD | TBD | TBD | TBD | TBD |

*학습 완료 후 결과가 업데이트됩니다.*

---

## 인용

연구에서 이 코드를 사용하는 경우 다음과 같이 인용해 주세요:

```bibtex
@inproceedings{chamnet2024,
  title={ChamNet: Multi-Task Learning for Agricultural Scene Understanding},
  author={Your Name},
  year={2024}
}
```

---

## 라이선스

이 프로젝트는 MMSegmentation과 MMDeploy를 기반으로 합니다. 각각의 라이선스를 참조하세요.

---

## 감사의 글

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - 프레임워크 기반
- [MMDeploy](https://github.com/open-mmlab/mmdeploy) - 모델 배포 툴킷
- [SegFormer](https://github.com/NVlabs/SegFormer) - 백본 아키텍처

---

## 문의

질문이나 이슈가 있는 경우 이 저장소에 이슈를 등록해 주세요.
