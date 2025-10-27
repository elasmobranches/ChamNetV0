# ESANet 멀티태스크 학습 코드 분석 가이드

## 📚 개요
이 문서는 `train_esanet_mtl_Uncertain.py` 파일의 모든 함수와 클래스를  상세히 설명합니다. 이 코드는 **세그멘테이션(이미지 분할)**과 **깊이 추정**을 동시에 학습하는 멀티태스크 모델을 구현합니다.

**⚠️ 중요**: 이 코드는 **불확실성 가중치(Uncertainty Weighting)** 방식을 사용하며, DWA(Dynamic Weight Average)는 사용하지 않습니다.

---

## 🏗️ 전체 아키텍처

### 핵심 개념
- **멀티태스크 학습**: 하나의 모델로 여러 작업을 동시에 수행
- **ESANet**: RGB와 깊이 정보를 분리해서 처리하는 효율적인 네트워크
- **불확실성 가중치**: 각 태스크의 학습 난이도에 따라 자동으로 가중치 조정

---

## 📦 1. 상수 및 설정

### `NUM_CLASSES = 7`
```python
NUM_CLASSES = 7
```
**역할**: 세그멘테이션에서 분류할 클래스 수
**필요한 이유**: 
- 모델이 몇 개의 클래스를 구분해야 하는지 알려줌
- 출력 레이어의 크기를 결정

### `ID2LABEL` 딕셔너리
```python
ID2LABEL: Dict[int, str] = {
    0: "background",    # 배경
    1: "chamoe",        # 참외
    2: "heatpipe",      # 히트파이프
    3: "path",          # 경로
    4: "pillar",        # 기둥
    5: "topdownfarm",   # 상하농장
    6: "unknown",       # 미분류
}
```
**역할**: 숫자 ID를 사람이 읽을 수 있는 클래스명으로 변환
**필요한 이유**: 
- 시각화할 때 클래스 이름을 표시
- 결과 해석을 쉽게 만듦

---

## 🔧 2. 유틸리티 함수들

### `set_global_determinism(seed: int = 42)`
```python
def set_global_determinism(seed: int = 42) -> None:
```
**역할**: 실험의 재현성을 보장하기 위해 모든 랜덤 시드 설정
**필요한 이유**: 
- 같은 코드를 여러 번 실행해도 동일한 결과가 나오도록 보장
- 실험 결과의 신뢰성 확보
- 버그 디버깅 시 일관된 결과 제공

**설정하는 것들**:
- Python 내장 random 모듈
- NumPy 랜덤 시드
- PyTorch CPU/CUDA 시드
- cuDNN 결정론적 모드

### `dataloader_worker_init_fn(worker_id: int)`
```python
def dataloader_worker_init_fn(worker_id: int) -> None:
```
**역할**: DataLoader의 각 워커 프로세스에 일관된 시드 설정
**필요한 이유**: 
- 멀티프로세싱으로 데이터를 로드할 때 각 워커가 다른 시드를 가지도록 함
- 데이터 셔플링의 재현성 보장

---

## 📊 3. 데이터셋 클래스

### `RGBDepthMultiTaskDataset`
```python
class RGBDepthMultiTaskDataset(Dataset):
```
**역할**: RGB 이미지, 세그멘테이션 마스크, 깊이 맵을 함께 로드하는 데이터셋
**필요한 이유**: 
- 멀티태스크 학습에 필요한 모든 데이터를 한 번에 제공
- 데이터 전처리와 정규화를 자동화

#### 주요 메서드:

##### `__init__()`
```python
def __init__(self, images_dir, masks_dir, depth_dir, image_size, mean, std, is_train=False):
```
**역할**: 데이터셋 초기화
**매개변수**:
- `images_dir`: RGB 이미지가 있는 폴더
- `masks_dir`: 세그멘테이션 마스크가 있는 폴더
- `depth_dir`: 깊이 맵이 있는 폴더
- `image_size`: 모델이 받을 이미지 크기
- `mean`, `std`: 이미지 정규화용 평균과 표준편차

##### `__getitem__(idx)`
```python
def __getitem__(self, idx: int):
```
**역할**: 인덱스에 해당하는 데이터 샘플을 로드하고 전처리
**반환값**: (RGB 이미지, 깊이 맵, 세그멘테이션 마스크, 깊이 타겟, 파일명)

**처리 과정**:
1. 파일 경로 생성 및 존재 확인
2. PIL 이미지로 로드
3. 깊이 데이터 정규화 (0-1 범위)
4. 모든 이미지를 모델 입력 크기로 리사이즈
5. PyTorch 텐서로 변환 및 정규화

### `get_preprocessing_params(encoder_name: str)`
```python
def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
```
**역할**: ImageNet 표준 정규화 파라미터 반환
**필요한 이유**: 
- 사전훈련된 모델과 호환되도록 표준 정규화 사용
- `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## 📈 4. 메트릭 누적기

### `MetricsAccumulator`
```python
class MetricsAccumulator:
```
**역할**: GPU에서 메트릭을 효율적으로 누적하고 에폭 종료 시 평균 계산
**필요한 이유**: 
- 스텝마다 CPU로 전송하면 오버헤드가 큼
- GPU에서 직접 연산하여 성능 향상
- 메모리 사용량 최적화

#### 주요 메서드:

##### `update(**kwargs)`
```python
def update(self, **kwargs):
```
**역할**: 새로운 메트릭 값들을 누적기에 추가
**처리**: GPU에서 유지하며 그래디언트 분리

##### `compute_mean()`
```python
def compute_mean(self) -> Dict[str, float]:
```
**역할**: 누적된 모든 메트릭의 평균을 계산
**최적화**: CPU로 한 번만 이동하여 효율성 향상

##### `reset()`
```python
def reset(self):
```
**역할**: 누적기를 리셋하고 메모리 해제

---

## ⚖️ 5. 동적 가중치 평균 (DWA) - 현재 비활성화됨

### `DynamicWeightAverage`
```python
class DynamicWeightAverage:
```
**역할**: 멀티태스크 학습에서 각 태스크의 상대적 손실 감소율을 기반으로 가중치를 자동 조정
**⚠️ 현재 상태**: 이 코드에서는 **사용되지 않음** (불확실성 가중치 방식 사용)

**필요한 이유** (DWA 사용 시): 
- 태스크 간 불균형 해결
- 모든 태스크가 균등하게 학습되도록 함
- 수동 가중치 조정의 어려움 해결

#### 주요 메서드:

##### `update_weights(current_losses)`
```python
def update_weights(self, current_losses: List[float]) -> List[float]:
```
**역할**: 상대적 손실 감소율을 기반으로 태스크 가중치 업데이트
**작동 원리**:
1. 각 태스크의 손실 감소율 계산
2. 감소율이 높은 태스크에 더 높은 가중치 부여
3. Softmax 함수로 가중치 정규화

**수치 안정성**: LogSumExp 트릭 사용으로 오버플로우 방지

---

## 🎯 6. 불확실성 가중치 시스템

### `LightningESANetMTL` 클래스의 불확실성 가중치 관련 속성들
```python
# 불확실성 가중치 관련 파라미터들
self.use_uncertainty_weighting = use_uncertainty_weighting
self.use_dwa_weighting = use_dwa_weighting
self.seg_loss_weight = seg_loss_weight
self.depth_loss_weight = depth_loss_weight
```

**역할**: 각 태스크의 학습 난이도에 따라 자동으로 가중치를 조정하는 시스템
**필요한 이유**: 
- 세그멘테이션과 깊이 추정의 학습 난이도가 다름
- 수동 가중치 조정의 어려움 해결
- Kendall et al. 논문의 수식 기반 자동 조정

**작동 원리**:
1. 각 태스크의 불확실성(σ)을 학습 가능한 파라미터로 설정
2. 손실 함수: `L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log(σ₁) + log(σ₂)`
3. 불확실성이 높은 태스크(학습이 어려운 태스크)에 더 낮은 가중치 부여

---

## 🎯 7. 손실 함수들

### `SILogLoss`
```python
class SILogLoss(nn.Module):
```
**역할**: 깊이 추정을 위한 스케일 불변 로그 손실
**필요한 이유**: 
- 깊이 값의 절대적 크기에 관계없이 상대적 오차에 집중
- 로그 공간에서 계산하여 더 안정적인 학습

**수식**: `SILog = (1/n) * Σ(log(pred) - log(target))² - λ * (1/n) * Σ(log(pred) - log(target))²`

### `L1DepthLoss`
```python
class L1DepthLoss(nn.Module):
```
**역할**: 유효 마스크 처리가 포함된 깊이 추정용 L1 손실
**필요한 이유**: 
- 절대 오차의 평균을 계산
- 유효하지 않은 픽셀은 손실 계산에서 제외

---

## 🧠 8. 메인 모델 클래스

### `ESANetMultiTask`
```python
class ESANetMultiTask(nn.Module):
```
**역할**: 세그멘테이션과 깊이 추정을 위한 ESANet 기반 멀티태스크 모델
**필요한 이유**: 
- RGB와 Depth를 분리된 입력으로 받는 효율적인 구조
- 사전훈련된 가중치 로딩 지원
- 작은 배치 크기에 대한 최적화

#### 주요 메서드:

##### `_fix_batchnorm_for_small_batches()`
```python
def _fix_batchnorm_for_small_batches(self):
```
**역할**: ESANet 내부의 BatchNorm을 작은 배치 크기에 맞게 GroupNorm으로 교체
**필요한 이유**: 
- 작은 배치에서 BatchNorm은 통계량 추정이 불안정
- GroupNorm으로 교체하여 안정적인 학습 보장

**동적 그룹 수 설정**:
- 32채널 이상: 32그룹
- 16채널 이상: 16그룹
- 8채널 이상: 8그룹
- 그 외: 채널수/2 그룹

##### `_load_pretrained_weights_safe()`
```python
def _load_pretrained_weights_safe(self, pretrained_path: str):
```
**역할**: 안전한 사전 학습된 가중치 로드
**필요한 이유**: 
- 호환되는 가중치만 로드
- 모델 아키텍처 불일치 시 안전하게 처리
- 구체적인 에러 메시지 제공

**에러 처리**:
- `FileNotFoundError`: 파일이 없을 때
- `RuntimeError`: 모델 구조 불일치 시
- 기타 예외는 재발생

##### `forward(rgb, depth)`
```python
def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
**역할**: 모델의 순전파 연산
**반환값**: (세그멘테이션 로짓, 깊이 예측)

**처리 과정**:
1. ESANet 순전파 (40개 클래스 출력)
2. 40개 클래스에서 7개 클래스로 변환
3. 깊이 헤드 순전파
4. 시그모이드 적용하여 깊이를 [0, 1] 범위로 제한

---

## 📊 9. 메트릭 계산 함수

### `compute_depth_metrics()`
```python
def compute_depth_metrics(pred, target, eps=1e-6) -> Dict[str, torch.Tensor]:
```
**역할**: 표준 깊이 추정 메트릭 계산
**필요한 이유**: 
- 깊이 추정 성능을 정량적으로 평가
- 다양한 메트릭으로 모델 성능 종합 평가

**계산하는 메트릭**:
- `abs_rel`: 절대 상대 오차
- `sq_rel`: 제곱 상대 오차
- `rmse`: 제곱근 평균 제곱 오차
- `rmse_log`: 로그 공간에서의 RMSE
- `delta1, delta2, delta3`: 임계값 메트릭

---

## ⚡ 10. PyTorch Lightning 모듈

### `LightningESANetMTL`
```python
class LightningESANetMTL(pl.LightningModule):
```
**역할**: ESANet 멀티태스크 학습을 위한 PyTorch Lightning 모듈
**필요한 이유**: 
- 손실/메트릭 로깅, 체크포인트, 학습/검증/테스트 루프 관리
- 불확실성 가중치/DWA 등 멀티태스크 손실 균형화 전략 지원
- 자동 최적화 및 스케줄링

#### 주요 메서드:

##### `_compute_class_iou()`
```python
def _compute_class_iou(self, tp, fp, fn, prefix=""):
```
**역할**: 공통 IoU 계산 헬퍼 함수
**필요한 이유**: 
- 클래스별 IoU 계산 코드 중복 제거
- 코드 유지보수성 향상
- validation과 test 단계에서 클래스별 IoU 로깅

**현재 사용**: validation과 test 에폭 종료 시 클래스별 IoU를 계산하고 로깅

##### `_save_visuals_pil()`
```python
def _save_visuals_pil(self, imgs, seg_gts, seg_preds, depth_gts, depth_preds, stage, save_count, filenames, colorize_seg, colorize_depth):
```
**역할**: PIL 기반 시각화 결과 저장
**필요한 이유**: 
- torchvision.utils 대신 PIL 사용
- 메모리 누수 방지를 위한 컨텍스트 매니저 사용
- 순수 numpy 기반 컬러맵 구현

##### `_compute_loss()`
```python
def _compute_loss(self, seg_logits, depth_pred, seg_target, depth_target) -> Tuple[torch.Tensor, Dict]:
```
**역할**: 불확실성 가중치 또는 DWA를 사용한 멀티태스크 손실 계산
**필요한 이유**: 
- 각 태스크의 학습 난이도에 따라 자동으로 가중치 조정
- 수동 가중치 조정의 어려움 해결

**가중치 전략**:
1. **불확실성 가중치**: Kendall et al. 논문의 수식 사용
2. **DWA 가중치**: 손실 감소율 기반 자동 조정
3. **수동 가중치**: 고정된 가중치 사용

##### `training_step()`
```python
def training_step(self, batch, batch_idx):
```
**역할**: 학습 스텝: 순전파, 손실 계산, 메트릭 계산, 로깅
**필요한 이유**: 
- PyTorch Lightning의 학습 루프 자동화
- 메트릭 로깅 및 시각화

##### `validation_step()`
```python
def validation_step(self, batch, batch_idx):
```
**역할**: 검증 스텝: 모델 성능 평가
**필요한 이유**: 
- 학습 중 모델 성능 모니터링
- 조기 종료 및 체크포인트 저장 기준
- torchmetrics를 사용한 정확한 mIoU 누적

**주요 개선사항**:
- torchmetrics의 `val_iou`를 사용하여 정확한 에폭별 mIoU 계산
- 배치별 부정확한 mIoU 로깅 제거
- FPS 계산을 단일 이미지 기준으로 개선

##### `test_step()`
```python
def test_step(self, batch, batch_idx):
```
**역할**: 테스트 스텝: 최종 모델 성능 평가
**필요한 이유**: 
- 학습 완료 후 모델의 최종 성능 측정
- 시각화 결과 저장
- torchmetrics를 사용한 정확한 mIoU 누적

**주요 개선사항**:
- torchmetrics의 `test_iou`를 사용하여 정확한 에폭별 mIoU 계산
- 배치별 부정확한 mIoU 로깅 제거
- FPS 계산을 단일 이미지 기준으로 개선

##### `on_train_epoch_end()`
```python
def on_train_epoch_end(self) -> None:
```
**역할**: 학습 에폭 종료 시 처리
**필요한 이유**: 
- 에폭 레벨 메트릭 계산 및 로깅
- 메트릭 누적기 리셋

##### `on_validation_epoch_end()`
```python
def on_validation_epoch_end(self) -> None:
```
**역할**: 검증 에폭 종료 시 처리
**필요한 이유**: 
- 검증 성능 모니터링
- 조기 종료 판단
- torchmetrics를 사용한 정확한 에폭별 mIoU 계산

**주요 개선사항**:
- torchmetrics의 `val_iou.compute()`로 정확한 에폭별 mIoU 계산
- 클래스별 IoU 계산 및 로깅 활성화
- 메모리 효율적인 메트릭 누적기 사용

##### `on_test_epoch_end()`
```python
def on_test_epoch_end(self) -> None:
```
**역할**: 테스트 에폭 종료 시 처리
**필요한 이유**: 
- 최종 성능 평가
- 클래스별 IoU 계산
- torchmetrics를 사용한 정확한 에폭별 mIoU 계산

**주요 개선사항**:
- torchmetrics의 `test_iou.compute()`로 정확한 에폭별 mIoU 계산
- 클래스별 IoU 계산 및 로깅 활성화
- 수동 계산 방식에서 torchmetrics 방식으로 전환

##### `configure_optimizers()`
```python
def configure_optimizers(self):
```
**역할**: 옵티마이저와 스케줄러 설정
**필요한 이유**: 
- AdamW 옵티마이저와 CosineAnnealingLR 스케줄러 설정
- 학습률 자동 조정

---

## 🔧 11. 데이터셋 빌더

### `build_esanet_datasets()`
```python
def build_esanet_datasets(dataset_root: Path, image_size: Tuple[int, int]):
```
**역할**: ESANet 멀티태스크 학습을 위한 데이터셋 구축
**필요한 이유**: 
- 학습/검증/테스트 데이터셋을 일관된 방식으로 생성
- 전처리 파이프라인 일관성 유지

**데이터셋 구조**:
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

---

## 🔄 12. 콜레이트 함수

### `collate_esanet_batch()`
```python
def collate_esanet_batch(batch):
```
**역할**: ESANet 배치 데이터를 안전하게 스택하는 커스텀 콜레이트 함수
**필요한 이유**: 
- 메모리 리사이즈 오류 방지
- 배치 크기 동적 처리
- 파일명 정보 보존

**주요 기능**:
- `contiguous()` 호출로 독립적 스토리지 보장
- 다양한 배치 크기에 대응
- 테스트 시 파일명 정보 보존

---

## 🚀 13. 메인 함수

### `main()`
```python
def main() -> None:
```
**역할**: ESANet 멀티태스크 학습의 메인 함수
**필요한 이유**: 
- 전체 학습 파이프라인을 통합 관리
- 설정 파일 로드 및 검증
- 데이터셋 구축 및 DataLoader 생성
- 모델 초기화 및 사전훈련 가중치 로드
- PyTorch Lightning Trainer 설정 및 학습 실행

**처리 순서**:
1. 전역 시드 설정 및 재현성 보장
2. 설정 파일 로드 (YAML 우선, JSON 폴백)
3. 데이터셋 경로 검증 및 구축
4. 모델 및 옵티마이저 초기화
5. 콜백 및 로거 설정
6. 학습/검증/테스트 실행
7. 결과 요약 및 저장

---

## 📋 14. 설정 파일 파싱

### `parse_args()`
```python
def parse_args():
```
**역할**: 명령행 인수를 파싱하여 설정 파일 경로와 옵션을 반환
**필요한 이유**: 
- 사용자가 명시적으로 설정 파일을 지정하도록 강제
- 기본 설정 파일 생성 옵션 제공

---

## 🎨 15. 시각화 함수들

### `_maybe_save_visuals()`
```python
def _maybe_save_visuals(self, rgb, depth, seg_masks, depth_target, seg_logits, depth_pred, stage, batch_idx, filenames=None):
```
**역할**: 시각화 결과 저장
**필요한 이유**: 
- 학습 과정에서 모델의 예측 결과를 시각적으로 확인
- 디버깅 및 성능 분석에 유용

### `_save_visuals_pil()`
```python
def _save_visuals_pil(self, imgs, seg_gts, seg_preds, depth_gts, depth_preds, stage, save_count, filenames, colorize_seg, colorize_depth):
```
**역할**: PIL 기반 시각화 저장
**필요한 이유**: 
- torchvision.utils 대신 PIL 사용
- 메모리 누수 방지를 위한 컨텍스트 매니저 사용

### `colorize_seg()` (내부 함수)
```python
def colorize_seg(label_hw: torch.Tensor) -> np.ndarray:
```
**역할**: 세그멘테이션 마스크를 컬러 이미지로 변환
**필요한 이유**: 
- 각 클래스별로 고유한 색상 할당
- 시각화를 위한 컬러맵 적용

### `colorize_depth()` (내부 함수)
```python
def colorize_depth(depth_hw: torch.Tensor) -> np.ndarray:
```
**역할**: 깊이 맵을 컬러 이미지로 변환 (viridis 컬러맵)
**필요한 이유**: 
- 깊이 정보를 직관적으로 시각화
- 순수 numpy 기반 구현으로 의존성 최소화

---

## 🔧 15. 최근 개선사항 (2024년 업데이트)

### torchmetrics 통합
**목적**: 정확한 mIoU 계산을 위한 검증된 라이브러리 사용
**적용된 변경사항**:
- `validation_step`과 `test_step`에서 torchmetrics의 IoU 메트릭 사용
- 배치별 부정확한 mIoU 로깅 제거
- 에폭별 정확한 mIoU 계산으로 전환

### FPS 계산 개선
**기존**: `fps = 1.0 / dt` (전체 배치 기준)
**개선**: `fps = 1.0 / (dt / rgb.shape[0])` (단일 이미지 기준)
**장점**: 더 직관적이고 일관된 성능 측정

### 클래스별 IoU 로깅 활성화
**변경사항**: validation과 test 단계에서 클래스별 IoU 계산 및 로깅
**구현**: `_compute_class_iou()` 헬퍼 함수 활용
**장점**: 세부적인 성능 분석 가능

### 메모리 효율성 개선
**PIL 이미지 처리**: 컨텍스트 매니저 사용으로 메모리 누수 방지
**BatchNorm 최적화**: 작은 배치 크기에 대한 GroupNorm 동적 변환
**수치 안정성**: LogSumExp 트릭으로 오버플로우 방지

### 에러 처리 강화
**사전훈련 가중치 로딩**: 구체적인 에러 메시지와 안전한 폴백
**JSON 설정 파일**: YAML 우선, JSON 폴백 시 경고 메시지
**호환성 검사**: 모델 아키텍처 불일치 시 안전한 처리

---

## 🔍 16. 핵심 개념 정리

### 멀티태스크 학습이란?
- **정의**: 하나의 모델로 여러 작업을 동시에 수행하는 학습 방법
- **장점**: 
  - 계산 자원 효율성
  - 태스크 간 상호 보완적 학습
  - 공통 특징 추출

### 불확실성 가중치란?
- **정의**: 각 태스크의 학습 난이도에 따라 자동으로 가중치를 조정하는 방법
- **수식**: `L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log(σ₁) + log(σ₂)`
- **장점**: 수동 가중치 조정의 어려움 해결

### DWA (Dynamic Weight Average)란?
- **정의**: 각 태스크의 상대적 손실 감소율을 기반으로 가중치를 자동 조정
- **작동 원리**: 감소율이 높은 태스크에 더 높은 가중치 부여
- **장점**: 태스크 간 불균형 해결

### torchmetrics란?
- **정의**: PyTorch 기반의 검증된 메트릭 계산 라이브러리
- **장점**: 
  - 정확하고 효율적인 메트릭 계산
  - GPU 최적화된 구현
  - 자동 배치 누적 및 에폭별 집계
- **사용법**: 
  - `update()`: 배치별로 메트릭 누적
  - `compute()`: 에폭별 정확한 값 계산
  - `reset()`: 다음 에폭을 위해 리셋

---

## 🎯 17. 학습 과정 요약

1. **데이터 준비**: RGB 이미지, 세그멘테이션 마스크, 깊이 맵 로드
2. **모델 초기화**: ESANet 기반 멀티태스크 모델 생성
3. **손실 계산**: 불확실성 가중치 또는 DWA 사용
4. **메트릭 계산**: IoU, AbsRel 등 성능 지표 계산
5. **최적화**: AdamW 옵티마이저로 가중치 업데이트
6. **검증**: 검증 데이터로 성능 평가
7. **테스트**: 최종 성능 측정 및 시각화

---

## 💡 18. 코드의 장점

1. **모듈화**: 각 기능이 독립적인 클래스/함수로 분리
2. **재사용성**: 다른 프로젝트에서도 활용 가능
3. **확장성**: 새로운 태스크나 손실 함수 쉽게 추가
4. **안정성**: 에러 처리 및 예외 상황 대응
5. **성능**: GPU 최적화 및 메모리 효율성 고려
6. **가독성**: 상세한 주석과 명확한 변수명

---

## 🚀 19. 실행 방법

```bash
# 기본 실행
python train_esanet_mtl_revise_uncertain.py --config config.yaml

# 기본 설정 파일 생성
python train_esanet_mtl_revise_uncertain.py --create-default-config
```

---

## 📚 20. 추가 학습 자료

- **PyTorch Lightning**: https://pytorch-lightning.readthedocs.io/
- **torchmetrics**: https://torchmetrics.readthedocs.io/
- **멀티태스크 학습**: https://arxiv.org/abs/1705.07115
- **불확실성 가중치**: https://arxiv.org/abs/1705.07115
- **DWA**: https://arxiv.org/abs/1803.10704

---

이 문서를 통해 코드의 각 부분이 왜 필요한지, 어떻게 작동하는지 이해할 수 있을 것입니다. 궁금한 부분이 있다면 언제든 질문해주세요! 🎓
