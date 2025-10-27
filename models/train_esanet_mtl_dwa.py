import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import csv

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PILImage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset



# torchmetrics 라이브러리 import (효율적인 메트릭 계산을 위해)
# IoU, MSE 등의 메트릭을 GPU에서 효율적으로 계산
try:
    from torchmetrics import JaccardIndex, MeanSquaredError, MeanAbsoluteError
    TORCHMETRICS_AVAILABLE = True
    print("✅ torchmetrics를 성공적으로 import했습니다.")
except ImportError:
    print("⚠️ torchmetrics가 설치되지 않았습니다. pip install torchmetrics를 실행하세요.")
    TORCHMETRICS_AVAILABLE = False

# PyTorch Lightning 관련 import
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time

# PIL 기반 시각화 사용 (torchvision.utils 제거됨)

# 학습 곡선 저장을 위한 matplotlib import
import matplotlib.pyplot as plt

# ESANet 모델 import
# ESANet은 RGB와 Depth를 분리된 입력으로 받는 효율적인 멀티태스크 아키텍처
sys.path.append('/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet')
try:
    from src.models.model import ESANet
    ESANET_AVAILABLE = True
    print("✅ ESANet 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ ESANet 모델 import 실패: {e}")
    ESANET_AVAILABLE = False


# ============================================================================
# 상수 정의
# ============================================================================
# 세그멘테이션 클래스 수 (배경 + 6개 객체 클래스)
NUM_CLASSES = 7

# 클래스 ID와 라벨명 매핑
# 각 클래스는 고유한 색상으로 시각화됨
ID2LABEL: Dict[int, str] = {
    0: "background",    # 배경 (검은색)
    1: "chamoe",        # 참외 (노란색)
    2: "heatpipe",      # 히트파이프 (빨간색)
    3: "path",          # 경로 (초록색)
    4: "pillar",        # 기둥 (파란색)
    5: "topdownfarm",   # 상하농장 (자홍색)
    6: "unknown",       # 미분류 (회색)
}


# ============================================================================
# 재현성 및 결정론 유틸리티
# ============================================================================
def set_global_determinism(seed: int = 42) -> None:
    """
    실험의 재현성을 보장하기 위해 모든 랜덤 시드를 설정합니다.
    
    Args:
        seed: 시드 값 (기본값: 42)
    
    이 함수는 다음을 설정합니다:
    - Python 내장 random 모듈 시드
    - NumPy 랜덤 시드
    - PyTorch CPU 시드
    - PyTorch CUDA 시드 (모든 GPU)
    - cuDNN 결정론적 모드
    - PyTorch 결정론적 알고리즘 (지원되지 않는 연산은 경고만 출력)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN 최적화 모드 설정 (성능 우선)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def dataloader_worker_init_fn(worker_id: int) -> None:
    """
    DataLoader의 각 워커 프로세스에 일관된 시드를 설정합니다.
    
    Args:
        worker_id: 워커 ID (PyTorch에서 자동으로 할당)
    
    각 워커는 서로 다른 시드를 가지지만, 재시작 시에도 동일한 시드를 사용합니다.
    이를 통해 데이터 셔플링과 증강의 재현성을 보장합니다.
    """
    # torch.initial_seed()는 이미 워커마다 다른 값을 가짐
    # 이를 numpy/python 시드와 동기화
    worker_seed = torch.initial_seed() % 2**32
    import random
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# ============================================================================
# RGB+Depth 멀티태스크 데이터셋 (분리된 입력)
# ============================================================================
class RGBDepthMultiTaskDataset(Dataset):
    """
    세그멘테이션과 깊이 추정을 위한 RGB/Depth 멀티태스크 데이터셋입니다.
    
    주요 특징:
    - ESANet 아키텍처에 맞게 RGB와 Depth를 분리된 입력으로 제공
    - albumentations를 활용한 고급 데이터 증강 지원
    - 학습 시 색상 증강(RGB만)과 기하학적 증강(RGB/Depth/Mask 모두) 분리 적용
    - 깊이 데이터의 정규화 및 전처리 자동화
    
    반환 데이터:
    - RGB 텐서: [3, H, W] 정규화된 RGB 이미지
    - Depth 텐서: [1, H, W] 정규화된 깊이 맵
    - 세그멘테이션 마스크: [H, W] 클래스 라벨
    - 깊이 정답: [H, W] 깊이 타겟 (시각화용)
    - 파일명: 원본 이미지 파일명
    """
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        depth_dir: Path,
        image_size: Tuple[int, int],
        mean: List[float],
        std: List[float],
        is_train: bool = False,
    ) -> None:
        """
        데이터셋 초기화
        
        Args:
            images_dir: RGB 이미지 디렉토리 경로
            masks_dir: 세그멘테이션 마스크 디렉토리 경로
            depth_dir: 깊이 맵 디렉토리 경로
            image_size: 모델 입력 크기 (width, height)
            mean: ImageNet 정규화 평균값 [R, G, B]
            std: ImageNet 정규화 표준편차 [R, G, B]
            is_train: 학습 모드 여부 (현재는 사용되지 않음)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

        # 지원되는 이미지 파일 확장자로 파일 목록 생성
        # JPG, JPEG, PNG 형식 지원
        self.image_files: List[str] = [
            f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images in {images_dir}")

        print(f"📁 Dataset loaded: {len(self.image_files)} images from {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        인덱스에 해당하는 샘플을 로드하고 전처리/증강/텐서화하여 반환합니다.
        
        처리 순서:
        1) 파일 경로 생성 및 존재 확인 (RGB/Mask/Depth)
        2) PIL 이미지 로드 및 깊이 스케일 정규화 (0~1 범위)
        3) 기본 리사이즈 (모델 입력 크기로 조정)
        4) (학습 모드 && albumentations 활성) 증강 분리 적용
           - RGB 전용: 밝기/대비/가우시안노이즈/감마/HSV 등 (깊이/마스크는 제외)
           - 기하학적: 좌우반전/회전/스케일/엘라스틱 (RGB/Depth/Mask 동일 변환)
        5) 증강 이후 크기 보정 (배치 스택을 위해 HxW 고정)
        6) 텐서 변환 및 정규화, Depth 타깃 생성
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            tuple: (image[3,H,W], depth[1,H,W], mask[H,W], depth_target[H,W], filename)
                - image: 정규화된 RGB 이미지 텐서
                - depth: 정규화된 깊이 맵 텐서
                - mask: 세그멘테이션 마스크 텐서
                - depth_target: 깊이 타겟 텐서 (시각화용)
                - filename: 원본 파일명
        """
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        # 파일명에서 확장자 제거하고 stem 추출
        # 예: "image001.jpg" -> "image001"
        img_stem = Path(img_name).stem
        
        # 마스크 파일 경로: {stem}_mask.png
        # 예: "image001" -> "image001_mask.png"
        mask_name = f"{img_stem}_mask.png"
        mask_path = self.masks_dir / mask_name
        
        # 깊이 파일 경로: {stem}_depth.png  
        # 예: "image001" -> "image001_depth.png"
        depth_name = f"{img_stem}_depth.png"
        depth_path = self.depth_dir / depth_name

        # 파일 존재 확인 (필수 파일들이 없으면 에러 발생)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # PIL 이미지로 데이터 로드
        # RGB 이미지는 3채널로 변환, 마스크는 그레이스케일로 변환
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        depth_img = Image.open(depth_path)
        
        # 깊이 이미지를 정규화된 float 배열로 변환
        # 다양한 깊이 이미지 포맷 지원 (16비트, 8비트, 그레이스케일)
        if depth_img.mode == 'I;16':
            # 16비트 깊이 이미지: 0-65535 범위를 0-1로 정규화
            depth = np.array(depth_img, dtype=np.float32) / 65535.0
        elif depth_img.mode == 'I':
            # 32비트 정수 깊이 이미지: 최대값으로 정규화
            depth = np.array(depth_img, dtype=np.float32)
            if depth.max() > 1.0:
                depth = depth / depth.max()
        else:
            # 기타 포맷: 그레이스케일로 변환 후 0-255 범위를 0-1로 정규화 
            # 우리는 이거 사용함
            depth = np.array(depth_img.convert('L'), dtype=np.float32) / 255.0
        
        # 정규화된 깊이 배열을 PIL 이미지로 변환
        depth = Image.fromarray(depth)

        # 모든 이미지를 모델 입력 크기로 리사이즈
        # 모든 이미지를 모델 입력 크기로 리사이즈 (전처리 파이프라인 일관성 유지)
        # RGB와 깊이는 BILINEAR 보간, 마스크는 NEAREST 보간 사용
        image = TF.resize(image, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.NEAREST)
        depth = TF.resize(depth, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)

        # PIL 이미지를 PyTorch 텐서로 변환 (독립적이고 리사이즈 가능한 스토리지 보장)
        image = TF.to_tensor(image).contiguous().clone()  # [3, H, W] in [0, 1]
        depth_tensor = torch.from_numpy(np.array(depth, dtype=np.float32)).contiguous().clone()  # [H, W]
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).contiguous().clone()  # [H, W]

        # ImageNet 표준으로 RGB 이미지 정규화
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # ESANet은 RGB와 Depth를 분리된 텐서로 받음
        # 깊이 텐서에 채널 차원 추가: [H, W] -> [1, H, W]
        depth_tensor = depth_tensor.unsqueeze(0).contiguous()  # [1, H, W]
        
        # depth_target은 이미 contiguous()로 독립적 스토리지 보장됨
        depth_target = depth_tensor.squeeze(0)

        return image, depth_tensor, mask, depth_target, img_name  # RGB, Depth, Mask, Depth_target, filename


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    """
    인코더 전처리 정규화 파라미터를 반환합니다 (ImageNet 표준).
    
    Args:
        encoder_name: 인코더 이름 (현재는 사용되지 않지만 확장성을 위해 유지)
        
    Returns:
        tuple: (mean, std) 정규화 파라미터
            - mean: [0.485, 0.456, 0.406] (R, G, B 채널별 평균)
            - std:  [0.229, 0.224, 0.225] (R, G, B 채널별 표준편차)
    
    Note:
        ESANet은 일반적으로 ImageNet 사전훈련된 백본을 사용하므로
        ImageNet 표준 정규화 파라미터를 사용합니다.
    """
    # ESANet은 일반적으로 ImageNet normalization 사용
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]




# ============================================================================
# 메모리 효율적인 메트릭 누적기
# ============================================================================
class MetricsAccumulator:
    """
    GPU 상에서 메트릭을 누적하고 에폭 종료 시 한 번에 평균을 계산하는 메모리 효율적인 누적기입니다.
    
    사용 이유:
    - 스텝마다 CPU로 전송/집계를 반복하면 오버헤드가 크게 증가
    - 에폭 끝에서 한 번에 평균을 계산하여 성능/메모리 효율성 향상
    - GPU 메모리에서 직접 연산하여 데이터 전송 비용 최소화
    
    주요 특징:
    - GPU에서 메트릭 값을 누적 저장
    - 에폭 종료 시에만 CPU로 전송하여 평균 계산
    - 메모리 사용량 최적화를 위한 자동 리셋 기능
    """
    def __init__(self, device: torch.device):
        """
        메트릭 누적기 초기화
        
        Args:
            device: 계산 디바이스 (GPU/CPU)
        """
        self.device = device
        self.metrics = {}  # 메트릭별 값들을 저장하는 딕셔너리
        self.count = 0     # 업데이트 횟수 카운터
        
    def update(self, **kwargs):
        """
        새로운 메트릭 값들을 누적기에 추가합니다 (GPU에서 유지)
        
        Args:
            **kwargs: 메트릭 이름과 값의 키워드 인수
        """
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                # 텐서인 경우: 그래디언트 분리하고 GPU에 유지
                value = value.detach()
                if value.device != self.device:
                    value = value.to(self.device)
            else:
                # 스칼라인 경우: 텐서로 변환
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            
            # 메트릭별로 값들을 리스트에 저장
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.count += 1
    
    def compute_mean(self) -> Dict[str, float]:
        """
        누적된 모든 메트릭의 평균을 계산합니다 (CPU로 한 번만 이동)
        
        Returns:
            Dict[str, float]: 메트릭 이름과 평균값의 딕셔너리
        """
        result = {}
        for key, values in self.metrics.items():
            if values:
                # 모든 값을 스택하고 평균 계산
                stacked = torch.stack(values)
                mean_value = stacked.mean()
                result[key] = float(mean_value.cpu().item())
        return result
    
    def reset(self):
        """
        누적기를 리셋하고 메모리를 해제합니다
        """
        # 저장된 모든 텐서를 삭제하여 메모리 해제
        for key, values in self.metrics.items():
            if values:
                del values[:]  # 리스트 내용 삭제
        self.metrics.clear()
        self.count = 0


# ============================================================================
# 동적 가중치 평균 (DWA) 손실 가중치 조정
# ============================================================================
class DynamicWeightAverage:
    """
    멀티태스크 학습을 위한 동적 가중치 평균(Dynamic Weight Average, DWA) 클래스입니다.
    
    주요 기능:
    - 각 태스크의 상대적 손실 감소율을 기반으로 가중치를 자동 조정
    - 태스크 간 불균형을 해결하여 모든 태스크가 균등하게 학습되도록 함
    - 학습 과정에서 동적으로 가중치를 업데이트하여 안정적인 학습 보장
    
    작동 원리:
    1. 각 태스크의 손실 감소율을 계산
    2. 감소율이 높은 태스크에 더 높은 가중치 부여
    3. Softmax 함수로 가중치를 정규화하여 합이 1이 되도록 함
    """
    def __init__(self, num_tasks: int = 2, temperature: float = 2.0, window_size: int = 10):
        """
        DWA 초기화
        
        Args:
            num_tasks: 태스크 수 (기본값: 2, 세그멘테이션 + 깊이 추정)
            temperature: Softmax 온도 파라미터 (기본값: 2.0)
            window_size: 손실 히스토리 윈도우 크기 (기본값: 10)
        """
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.window_size = window_size
        
        # 각 태스크별 손실 히스토리 저장
        self.loss_history = [[] for _ in range(num_tasks)]
        # 초기 가중치: 모든 태스크에 동일한 가중치 (1/태스크수)
        self.weights = [1.0 / num_tasks] * num_tasks
        
    def update_weights(self, current_losses: List[float]) -> List[float]:
        """
        상대적 손실 감소율을 기반으로 태스크 가중치를 업데이트합니다.
        
        Args:
            current_losses: 각 태스크의 현재 손실 값 리스트
            
        Returns:
            List[float]: 각 태스크의 업데이트된 가중치
        """
        # 현재 손실값을 히스토리에 추가
        for i, loss in enumerate(current_losses):
            self.loss_history[i].append(loss)
            
        # 최근 히스토리만 유지 (슬라이딩 윈도우)
        for i in range(self.num_tasks):
            if len(self.loss_history[i]) > self.window_size:
                self.loss_history[i] = self.loss_history[i][-self.window_size:]
        
        # 상대적 감소율 계산
        if all(len(history) >= 2 for history in self.loss_history):
            decrease_rates = []
            for i in range(self.num_tasks):
                history = self.loss_history[i]
                # 윈도우에 대한 평균 감소율 계산
                if len(history) >= 2:
                    decreases = []
                    for j in range(1, len(history)):
                        if history[j-1] > 0:
                            decrease = (history[j-1] - history[j]) / history[j-1]
                            decreases.append(max(0, decrease))  # 양수 감소만 고려
                    
                    if decreases:
                        avg_decrease = sum(decreases) / len(decreases)
                        decrease_rates.append(avg_decrease)
                    else:
                        decrease_rates.append(0.0)
                else:
                    decrease_rates.append(0.0)
            
            # 온도를 사용한 Softmax로 가중치 계산 (LogSumExp 트릭으로 수치 안정성 개선)
            if any(rate > 0 for rate in decrease_rates):
                # 감소율 정규화
                max_rate = max(decrease_rates)
                if max_rate > 0:
                    normalized_rates = [rate / max_rate for rate in decrease_rates]
                else:
                    normalized_rates = [1.0] * self.num_tasks
                
                # 온도 스케일링 적용 및 LogSumExp 트릭 사용
                scaled_rates = torch.tensor(normalized_rates) / self.temperature
                max_scaled = scaled_rates.max()
                exp_rates = torch.exp(scaled_rates - max_scaled)  # 수치 안정성
                self.weights = (exp_rates / exp_rates.sum()).tolist()
            else:
                # 감소가 없으면 동일한 가중치 사용
                self.weights = [1.0 / self.num_tasks] * self.num_tasks
        else:
            # 충분한 히스토리가 없으면 동일한 가중치 사용
            self.weights = [1.0 / self.num_tasks] * self.num_tasks
        
        return self.weights.copy()
    
    def get_weights(self) -> List[float]:
        """
        현재 가중치를 반환합니다
        
        Returns:
            List[float]: 각 태스크의 현재 가중치
        """
        return self.weights.copy()


# ============================================================================
# 손실 함수들
# ============================================================================
class SILogLoss(nn.Module):
    """
    깊이 추정을 위한 스케일 불변 로그 손실(Scale-Invariant Logarithmic Loss)입니다.
    
    주요 특징:
    - 스케일 불변성: 깊이 값의 절대적 크기에 관계없이 상대적 오차에 집중
    - 로그 공간에서 계산: 깊이 값의 로그 차이를 기반으로 손실 계산
    - 분산 항: 로그 차이의 분산을 고려하여 더 안정적인 학습
    
    수식:
    SILog = (1/n) * Σ(log(pred) - log(target))² - λ * (1/n) * Σ(log(pred) - log(target))²
    """
    def __init__(self, lambda_variance: float = 0.85, eps: float = 1e-6):
        """
        SILog 손실 함수 초기화
        
        Args:
            lambda_variance: 분산 항의 가중치 (기본값: 0.85)
            eps: 수치 안정성을 위한 작은 값 (기본값: 1e-6)
        """
        super().__init__()
        self.lambda_variance = lambda_variance
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        SILog 손실 계산
        
        Args:
            pred: 예측된 깊이 맵 [B, 1, H, W]
            target: 정답 깊이 맵 [B, H, W] 또는 [B, 1, H, W]
            mask: 유효 픽셀 마스크 [B, H, W] 또는 [B, 1, H, W] (선택사항)
            
        Returns:
            torch.Tensor: SILog 손실 값
        """
        # 차원 정규화: [B, 1, H, W] -> [B, H, W]
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # 유효한 깊이 값에 대한 마스크 생성
        if mask is None:
            # 마스크가 없으면 유효한 깊이 값만 사용
            valid_mask = (target > self.eps) & (torch.isfinite(target))
        else:
            # 마스크가 있으면 마스크와 유효한 깊이 값 모두 고려
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask = (target > self.eps) & (torch.isfinite(target)) & (mask > 0.5)
        
        # 유효한 픽셀이 없으면 손실 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 마스크 적용: 유효한 픽셀만 사용
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # 로그 차이 계산: log(pred) - log(target)
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        
        # SILog 손실 계산: MSE - λ * (평균)²
        loss = (log_diff ** 2).mean() - self.lambda_variance * (log_diff.mean() ** 2)
        return loss


class L1DepthLoss(nn.Module):
    """
    유효 마스크 처리가 포함된 깊이 추정용 L1 손실 함수입니다.
    
    주요 특징:
    - L1 손실: 절대 오차의 평균을 계산
    - 유효 마스크 지원: 유효하지 않은 픽셀은 손실 계산에서 제외
    - 수치 안정성: 매우 작은 값으로 나누기 오류 방지
    """
    def __init__(self, eps: float = 1e-6):
        """
        L1 깊이 손실 함수 초기화
        
        Args:
            eps: 수치 안정성을 위한 작은 값 (기본값: 1e-6)
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L1 손실 계산
        
        Args:
            pred: 예측된 깊이 맵 [B, 1, H, W]
            target: 정답 깊이 맵 [B, H, W] 또는 [B, 1, H, W]
            mask: 유효 픽셀 마스크 [B, H, W] 또는 [B, 1, H, W] (선택사항)
            
        Returns:
            torch.Tensor: L1 손실 값
        """
        # 차원 정규화: [B, 1, H, W] -> [B, H, W]
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # 유효한 깊이 값에 대한 마스크 생성
        if mask is None:
            valid_mask = (target > self.eps) & (torch.isfinite(target))
        else:
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask = (target > self.eps) & (torch.isfinite(target)) & (mask > 0.5)
        
        # 유효한 픽셀이 없으면 손실 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 유효한 픽셀만 사용하여 L1 손실 계산
        pred = pred[valid_mask]
        target = target[valid_mask]
        return F.l1_loss(pred, target)


# ============================================================================
# ESANet 기반 멀티태스크 모델
# ============================================================================
class ESANetMultiTask(nn.Module):
    """
    세그멘테이션과 깊이 추정을 위한 ESANet 기반 멀티태스크 학습 모델입니다.
    
    아키텍처:
        - 공유 인코더: ESANet 인코더 (RGB+Depth 분리 입력)
        - 태스크별 헤드: 세그멘테이션 헤드 + 깊이 추정 헤드
    
    주요 특징:
    - RGB와 Depth를 분리된 입력으로 받는 효율적인 구조
    - 사전훈련된 가중치 로딩 지원
    - 작은 배치 크기에 대한 BatchNorm 최적화
    - 40개 클래스에서 7개 클래스로 변환하는 어댑터 포함
    """
    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        num_classes: int = 7,
        encoder_rgb: str = 'resnet34',
        encoder_depth: str = 'resnet34',
        encoder_block: str = 'NonBottleneck1D',
        pretrained_path: Optional[str] = None,
    ):
        """
        ESANet 멀티태스크 모델 초기화
        
        Args:
            height: 입력 이미지 높이 (기본값: 480)
            width: 입력 이미지 너비 (기본값: 640)
            num_classes: 세그멘테이션 클래스 수 (기본값: 7)
            encoder_rgb: RGB 인코더 백본 (기본값: 'resnet34')
            encoder_depth: Depth 인코더 백본 (기본값: 'resnet34')
            encoder_block: 인코더 블록 타입 (기본값: 'NonBottleneck1D')
            pretrained_path: 사전훈련된 가중치 경로 (선택사항)
        """
        super().__init__()
        
        if not ESANET_AVAILABLE:
            raise ImportError("ESANet 모델을 사용할 수 없습니다.")
        
        # ESANet 모델 초기화 (RGB+Depth 분리 입력용)
        # NYUv2 가중치와 호환되도록 40개 클래스로 초기화
        self.esanet = ESANet(
            height=height,
            width=width,
            num_classes=40,  # NYUv2 가중치와 호환을 위해 40개 클래스로 초기화
            encoder_rgb=encoder_rgb,
            encoder_depth=encoder_depth,
            encoder_block=encoder_block,
            pretrained_on_imagenet=False,  # ImageNet 사전 학습 비활성화
            activation='relu',
            encoder_decoder_fusion='add',
            context_module='ppm',
            fuse_depth_in_rgb_encoder='SE-add',
            upsampling='bilinear',
        )
        
        # ESANet 내부의 BatchNorm 설정 변경 (배치 크기 1 문제 해결)
        self._fix_batchnorm_for_small_batches()
        
        # 사전 학습된 가중치 로드 (선택사항) - 안전한 방식으로 처리
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading pretrained ESANet weights from {pretrained_path}")
            self._load_pretrained_weights_safe(pretrained_path)
        else:
            print("📝 No pretrained weights provided, training from scratch...")
        
        # 40개 클래스에서 7개 클래스로 변환하는 어댑터 추가
        # 1x1 컨볼루션으로 채널 수만 변경
        self.class_adapter = nn.Conv2d(40, num_classes, 1)
        
        # 깊이 추정 헤드: 향상된 CNN 구조 (단순화된 ASPP 접근법)
        self.depth_head = nn.Sequential(
            # 다중 스케일 특징 추출
            nn.Conv2d(40, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            
            # 확장 컨볼루션: 더 큰 수용 영역을 위한
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            
            # 최종 깊이 예측
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)  # 1채널 깊이 맵 출력
        )
        
        print(f"🔧 ESANet Multi-Task Architecture:")
        print(f"  - Input: RGB [B,3,H,W] + Depth [B,1,H,W] (separated)")
        print(f"  - Output: Segmentation ({num_classes} classes) + Depth (1 channel)")
        print(f"  - Encoder RGB: {encoder_rgb}")
        print(f"  - Encoder Depth: {encoder_depth}")
    
    def _fix_batchnorm_for_small_batches(self):
        """
        ESANet 내부의 BatchNorm을 작은 배치 크기에 맞게 수정합니다.
        
        작은 배치 크기에서 BatchNorm은 통계량 추정이 불안정하므로
        GroupNorm으로 교체하여 안정적인 학습을 보장합니다.
        """
        def fix_batchnorm_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    # BatchNorm을 GroupNorm으로 교체 (채널 수에 따른 동적 그룹 수 설정)
                    num_channels = child.num_features
                    if num_channels >= 32:
                        num_groups = 32
                    elif num_channels >= 16:
                        num_groups = 16
                    elif num_channels >= 8:
                        num_groups = 8
                    else:
                        num_groups = max(1, num_channels // 2)  # 최소 2채널당 1그룹
                    
                    group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features, 
                                            eps=child.eps, affine=child.affine)
                    
                    # 가중치와 바이어스 복사
                    if child.affine:
                        group_norm.weight.data = child.weight.data.clone()
                        group_norm.bias.data = child.bias.data.clone()
                    
                    # 모듈 교체
                    setattr(module, name, group_norm)
                else:
                    fix_batchnorm_recursive(child)
        
        fix_batchnorm_recursive(self.esanet)
        # BatchNorm을 GroupNorm으로 교체 완료
    
    def _load_pretrained_weights_safe(self, pretrained_path: str):
        """
        안전한 사전 학습된 가중치 로드 - 호환되는 가중치만 로드합니다.
        
        Args:
            pretrained_path: 사전훈련된 가중치 파일 경로
        """
        try:
            # .pth 파일 직접 로드
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 체크포인트 키 매핑 (다양한 체크포인트 형식 지원)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 현재 모델의 상태 딕셔너리 가져오기
            model_dict = self.esanet.state_dict()
            compatible_dict = {}
            
            # 조용히 가중치 호환성 확인
            compatible_count = 0
            incompatible_count = 0
            
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        compatible_dict[k] = v
                        compatible_count += 1
                    else:
                        # Shape mismatch는 조용히 스킵
                        incompatible_count += 1
                else:
                    # Missing key는 조용히 스킵
                    incompatible_count += 1
            
            # 호환되는 가중치만 로드
            if compatible_dict:
                model_dict.update(compatible_dict)
                self.esanet.load_state_dict(model_dict)
                print(f"✅ Loaded {compatible_count} pretrained weights ({incompatible_count} skipped)")
            else:
                print("📝 No compatible weights found, training from scratch...")
                
        except FileNotFoundError:
            print(f"⚠️ Pretrained file not found: {pretrained_path}")
            print("📝 Training from scratch...")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠️ Model architecture mismatch: {e}")
                print("📝 Training from scratch...")
            else:
                raise  # 다른 런타임 에러는 재발생
        except Exception as e:
            print(f"⚠️ Warning: Could not load pretrained weights: {e}")
            print("📝 Training from scratch...")
        
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ESANet 멀티태스크 모델의 순전파 연산
        
        Args:
            rgb: RGB 입력 텐서 [B, 3, H, W]
            depth: 깊이 입력 텐서 [B, 1, H, W]
        
        Returns:
            seg_logits: 세그멘테이션 로짓 [B, num_classes, H, W]
            depth_pred: 깊이 예측 [B, 1, H, W] (0-1 범위)
        """
        # ESANet 순전파 (40개 클래스 출력)
        esanet_output = self.esanet(rgb, depth)
        
        # ESANet이 훈련 모드에서 여러 출력을 반환하는 경우 처리
        if isinstance(esanet_output, tuple):
            esanet_features = esanet_output[0]  # 첫 번째 출력이 메인 segmentation 결과
        else:
            esanet_features = esanet_output
        
        # 40개 클래스에서 7개 클래스로 변환
        seg_logits = self.class_adapter(esanet_features)
        
        # 깊이 헤드 순전파 (40개 클래스 특징에서)
        depth_raw = self.depth_head(esanet_features)
        
        # 시그모이드 적용하여 깊이를 [0, 1] 범위로 제한
        depth_pred = torch.sigmoid(depth_raw)
        
        # 깊이 예측이 타겟과 동일한 형태 [B, H, W]가 되도록 보장
        if depth_pred.dim() == 4 and depth_pred.size(1) == 1:
            depth_pred = depth_pred.squeeze(1)
        
        return seg_logits, depth_pred


# ============================================================================
# 깊이 추정 메트릭
# ============================================================================
@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """
    표준 깊이 추정 메트릭을 계산합니다.
    
    Args:
        pred: 예측된 깊이 맵 [B, H, W]
        target: 정답 깊이 맵 [B, H, W]
        eps: 수치 안정성을 위한 작은 값
        
    Returns:
        Dict[str, torch.Tensor]: 메트릭 이름과 값의 딕셔너리
            - abs_rel: 절대 상대 오차
            - sq_rel: 제곱 상대 오차
            - rmse: 제곱근 평균 제곱 오차
            - rmse_log: 로그 공간에서의 RMSE
            - delta1, delta2, delta3: 임계값 메트릭
    """
    # 차원 정규화: [B, 1, H, W] -> [B, H, W]
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    # 유효한 픽셀 마스크 생성
    valid_mask = (target > eps) & (torch.isfinite(target)) & (torch.isfinite(pred))
    if valid_mask.sum() == 0:
        # 유효한 픽셀이 없으면 모든 메트릭을 0으로 반환
        return {
            "abs_rel": torch.tensor(0.0, device=pred.device),
            "sq_rel": torch.tensor(0.0, device=pred.device),
            "rmse": torch.tensor(0.0, device=pred.device),
            "rmse_log": torch.tensor(0.0, device=pred.device),
            "delta1": torch.tensor(0.0, device=pred.device),
            "delta2": torch.tensor(0.0, device=pred.device),
            "delta3": torch.tensor(0.0, device=pred.device),
        }
    
    # 유효한 픽셀만 사용
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # AbsRel: 절대 상대 오차
    abs_rel = (torch.abs(pred - target) / (target + eps)).mean()

    # SqRel: 제곱 상대 오차
    sq_rel = ((pred - target) ** 2 / (target + eps)).mean()

    # RMSE: 제곱근 평균 제곱 오차
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    
    # RMSE log: 로그 공간에서의 RMSE
    rmse_log = torch.sqrt(((torch.log(pred + eps) - torch.log(target + eps)) ** 2).mean())
    
    # 임계값 메트릭: 정확도 임계값 (δ < 1.25, δ < 1.25², δ < 1.25³)
    ratio = torch.maximum(pred / (target + eps), target / (pred + eps))
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()
    
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


# ============================================================================
# PyTorch Lightning 모듈
# ============================================================================
class LightningESANetMTL(pl.LightningModule):
    """
    ESANet 멀티태스크 학습을 위한 PyTorch Lightning 모듈입니다.
    
    주요 기능:
    - 손실/메트릭 로깅, 체크포인트, 학습/검증/테스트 루프 관리
    - 불확실성 가중치/DWA 등 멀티태스크 손실 균형화 전략 지원
    - 메모리 효율적인 메트릭 누적 및 시각화
    - 자동 최적화 및 스케줄링
    """
    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        num_classes: int = 7,
        encoder_rgb: str = 'resnet34',
        encoder_depth: str = 'resnet34',
        encoder_block: str = 'NonBottleneck1D',
        lr: float = 1e-4,
        scheduler_t_max: int = 1000,
        final_lr: float = 1e-5,
        loss_type: str = "silog",  # "silog" or "l1"
        seg_loss_weight: float = 1.0,
        depth_loss_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
        use_dwa_weighting: bool = True,
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
        pretrained_path: Optional[str] = None,
    ) -> None:
        """
        Lightning ESANet 멀티태스크 모듈 초기화
        
        Args:
            height: 입력 이미지 높이 (기본값: 480)
            width: 입력 이미지 너비 (기본값: 640)
            num_classes: 세그멘테이션 클래스 수 (기본값: 7)
            encoder_rgb: RGB 인코더 백본 (기본값: 'resnet34')
            encoder_depth: Depth 인코더 백본 (기본값: 'resnet34')
            encoder_block: 인코더 블록 타입 (기본값: 'NonBottleneck1D')
            lr: 학습률 (기본값: 1e-4)
            scheduler_t_max: 스케줄러 최대 에폭 (기본값: 1000)
            loss_type: 깊이 손실 함수 타입 (기본값: "silog")
            seg_loss_weight: 세그멘테이션 손실 가중치 (기본값: 1.0)
            depth_loss_weight: 깊이 손실 가중치 (기본값: 1.0)
            use_uncertainty_weighting: 불확실성 가중치 사용 여부 (기본값: True)
            use_dwa_weighting: DWA 가중치 사용 여부 (기본값: False)
            save_vis_dir: 시각화 저장 디렉토리 (기본값: "")
            vis_max: 최대 시각화 개수 (기본값: 4)
            save_root_dir: 저장 루트 디렉토리 (기본값: "")
            pretrained_path: 사전훈련된 가중치 경로 (선택사항)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # ESANet 멀티태스크 모델 초기화
        self.model = ESANetMultiTask(
            height=height,
            width=width,
            num_classes=num_classes,
            encoder_rgb=encoder_rgb,
            encoder_depth=encoder_depth,
            encoder_block=encoder_block,
            pretrained_path=pretrained_path,
        )
        
        # 손실 함수 설정
        # 세그멘테이션: Dice + Cross-Entropy 조합으로 개선
        try:
            import segmentation_models_pytorch as smp
            self.seg_dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            self.seg_ce_loss = nn.CrossEntropyLoss()
            self._use_dice = True
        except Exception:
            # smp 미존재 시 CE만 사용 (후속 설치 권장)
            self.seg_ce_loss = nn.CrossEntropyLoss()
            self._use_dice = False
        
        # 깊이 손실 함수 설정
        if loss_type == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = L1DepthLoss()
        
        # 손실 가중치 전략 설정
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_dwa_weighting = use_dwa_weighting
        
        if use_uncertainty_weighting:
            # 불확실성 기반 가중치: 학습 가능한 로그 분산 파라미터
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_depth = nn.Parameter(torch.zeros(1))
        elif self.use_dwa_weighting:
            # DWA 초기화: 2개 태스크 (세그멘테이션, 깊이)
            self.dwa = DynamicWeightAverage(num_tasks=2, temperature=2.0, window_size=10)
        else:
            # 수동 가중치 설정
            self.seg_loss_weight = seg_loss_weight
            self.depth_loss_weight = depth_loss_weight
        
        # 기본 설정 저장
        self.num_classes = num_classes
        self.lr = lr
        self.t_max = scheduler_t_max
        self.final_lr = float(final_lr)
        self.base_vis_dir = save_vis_dir
        self.vis_max = int(vis_max) if vis_max is not None else 0
        self.save_root_dir = save_root_dir
        
        # 최고 성능 추적은 ModelCheckpoint 콜백에서 처리
        
        # 학습 곡선 저장용 딕셔너리
        self.curves = {
            "train_loss": [],
            "val_loss": [],
            "train_seg_loss": [],
            "val_seg_loss": [],
            "train_depth_loss": [],
            "val_depth_loss": [],
            "train_miou": [],
            "val_miou": [],
            "train_abs_rel": [],
            "val_abs_rel": [],
            "train_sq_rel": [],
            "val_sq_rel": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_rmse_log": [],
            "val_rmse_log": [],
            "val_delta1": [],
        }
        
        # 메모리 효율적인 메트릭 누적기 (첫 번째 스텝에서 초기화)
        self._train_metrics_accumulator = None
        self._val_metrics_accumulator = None
        
        # 수동 계산을 위한 에폭 메트릭 초기화
        self._epoch_train_tp = None
        self._epoch_train_fp = None
        self._epoch_train_fn = None
        self._epoch_val_tp = None
        self._epoch_val_fp = None
        self._epoch_val_fn = None
        
        # 메트릭 누적기는 MetricsAccumulator로 처리
        
        # 효율적인 메트릭 계산을 위한 torchmetrics 설정
        if TORCHMETRICS_AVAILABLE:
            self.train_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
            self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
            self.test_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
        else:
            # torchmetrics 미사용 시 수동 계산으로 폴백
            self._epoch_train_tp = None
            self._epoch_train_fp = None
            self._epoch_train_fn = None
            self._epoch_val_tp = None
            self._epoch_val_fp = None
            self._epoch_val_fn = None
        
        # 적응형 정규화를 위한 최고 값들
        self.register_buffer("seg_best", torch.tensor(0.0))
        self.register_buffer("depth_best", torch.tensor(999.0))
        self.first_val_done = False
        # reset() 후 compute() 호출을 피하기 위한 마지막 에폭 레벨 val iou 저장
        self._last_val_epoch_iou = None
        
        # 시각화 파라미터 설정
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("vis_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("vis_std", torch.tensor(std).view(1, 3, 1, 1))
        
        # 세그멘테이션용 색상 팔레트
        self.register_buffer(
            "palette",
            torch.tensor([
                [0, 0, 0],       # 0 background (검은색)
                [255, 255, 0],   # 1 chamoe (노란색)
                [255, 0, 0],     # 2 heatpipe (빨간색)
                [0, 255, 0],     # 3 path (초록색)
                [0, 0, 255],     # 4 pillar (파란색)
                [255, 0, 255],   # 5 topdownfarm (자홍색)
                [128, 128, 128], # 6 unknown (회색)
            ], dtype=torch.uint8)
        )
        # 간단한 에폭 종료 요약 저장 (콘솔 출력용)
        self.logged_metrics = {}
    
    def _compute_class_iou(self, tp, fp, fn, prefix=""):
        """
        공통 IoU 계산 헬퍼 함수
        
        Args:
            tp: True Positive 텐서 [num_classes]
            fp: False Positive 텐서 [num_classes] 
            fn: False Negative 텐서 [num_classes]
            prefix: 로그 키 접두사 (예: "val", "test")
        """
        class_names = [
            "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
        ]
        for i, class_name in enumerate(class_names[: self.num_classes]):
            denom = (tp[i] + fp[i] + fn[i])
            iou_value = (tp[i] / denom) if denom > 0 else torch.tensor(0.0, device=tp.device)
            self.log(f"{prefix}_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모델 순전파
        
        Args:
            rgb: RGB 입력 텐서 [B, 3, H, W]
            depth: 깊이 입력 텐서 [B, 1, H, W]
            
        Returns:
            seg_logits: 세그멘테이션 로짓 [B, num_classes, H, W]
            depth_pred: 깊이 예측 [B, 1, H, W]
        """
        return self.model(rgb, depth)
    
    def _compute_loss(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                     seg_target: torch.Tensor, depth_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        불확실성 가중치 또는 DWA를 사용한 멀티태스크 손실 계산
        
        Args:
            seg_logits: 세그멘테이션 로짓 [B, num_classes, H, W]
            depth_pred: 깊이 예측 [B, 1, H, W]
            seg_target: 세그멘테이션 타겟 [B, H, W]
            depth_target: 깊이 타겟 [B, H, W]
            
        Returns:
            total_loss: 총 손실 값
            loss_dict: 손실 구성 요소들의 딕셔너리
        """
        
        # 태스크별 손실 계산
        if self._use_dice:
            # Dice + Cross-Entropy 조합 사용
            seg_dice = self.seg_dice_loss(seg_logits, seg_target)
            seg_ce = self.seg_ce_loss(seg_logits, seg_target)
            seg_loss = 0.7 * seg_dice + 0.3 * seg_ce
        else:
            # Cross-Entropy만 사용
            seg_dice = None
            seg_ce = self.seg_ce_loss(seg_logits, seg_target)
            seg_loss = seg_ce
        depth_loss = self.depth_loss_fn(depth_pred, depth_target)
        
        if self.use_uncertainty_weighting:
            # Kendall et al. (CVPR 2018): Multi-Task Learning Using Uncertainty
            # L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log(σ₁) + log(σ₂)
            
            # 안정성을 위한 클램핑 (exp 폭발 방지)
            log_var_seg = torch.clamp(self.log_var_seg, min=-5.0, max=5.0)
            log_var_depth = torch.clamp(self.log_var_depth, min=-5.0, max=5.0)
            
            precision_seg = torch.exp(-log_var_seg)      # 1/σ²
            precision_depth = torch.exp(-log_var_depth)  # 1/σ²
            
            weighted_seg_loss = 0.5 * precision_seg * seg_loss + 0.5 * log_var_seg
            weighted_depth_loss = 0.5 * precision_depth * depth_loss + 0.5 * log_var_depth
            
            total_loss = weighted_seg_loss + weighted_depth_loss
            
        elif self.use_dwa_weighting:
            # Dynamic Weight Average (Liu et al., CVPR 2019)
            # 손실 감소율 기반 자동 가중치 조정
            current_losses = [seg_loss.detach().item(), 
                            depth_loss.detach().item()]
            weights = self.dwa.update_weights(current_losses)
            
            weighted_seg_loss = weights[0] * seg_loss
            weighted_depth_loss = weights[1] * depth_loss
            total_loss = weighted_seg_loss + weighted_depth_loss
            
        else:
            # 수동 가중치: 고정된 가중치 사용
            weighted_seg_loss = self.seg_loss_weight * seg_loss
            weighted_depth_loss = self.depth_loss_weight * depth_loss
            total_loss = weighted_seg_loss + weighted_depth_loss
        
        # 손실 구성 요소들을 딕셔너리로 정리
        loss_dict = {
            "total": total_loss,
            "seg": seg_loss,
            "seg_dice": seg_dice if seg_dice is not None else torch.tensor(0.0, device=seg_logits.device),
            "seg_ce": seg_ce,
            "depth": depth_loss,
            "weighted_seg": weighted_seg_loss,
            "weighted_depth": weighted_depth_loss,
        }
        
        # 가중치 전략별 추가 정보 로깅
        if self.use_uncertainty_weighting:
            loss_dict["log_var_seg"] = self.log_var_seg
            loss_dict["log_var_depth"] = self.log_var_depth
            # 정밀도(불확실성) 값도 로깅
            loss_dict["precision_seg"] = torch.exp(-torch.clamp(self.log_var_seg, min=-5.0, max=5.0))
            loss_dict["precision_depth"] = torch.exp(-torch.clamp(self.log_var_depth, min=-5.0, max=5.0))
        elif self.use_dwa_weighting:
            # DWA 가중치 모니터링을 위한 로깅
            weights = self.dwa.get_weights()
            loss_dict["dwa_weight_seg"] = weights[0]
            loss_dict["dwa_weight_depth"] = weights[1]
        
        return total_loss, loss_dict

    @torch.no_grad()
    def _compute_metrics(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                        seg_target: torch.Tensor, depth_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        torchmetrics를 사용하여 두 태스크의 메트릭을 계산합니다 (사용 가능한 경우)
        
        Args:
            seg_logits: 세그멘테이션 로짓 [B, num_classes, H, W]
            depth_pred: 깊이 예측 [B, 1, H, W]
            seg_target: 세그멘테이션 타겟 [B, H, W]
            depth_target: 깊이 타겟 [B, H, W]
            
        Returns:
            Dict[str, torch.Tensor]: 메트릭 이름과 값의 딕셔너리
        """
        # 세그멘테이션 메트릭 계산
        prob = torch.softmax(seg_logits, dim=1)
        seg_pred = torch.argmax(prob, dim=1)
        
        # 간단한 정확도 계산
        correct = (seg_pred == seg_target).float()
        seg_acc = correct.mean()
        
        # 깊이 메트릭 (항상 계산)
        depth_metrics = compute_depth_metrics(depth_pred, depth_target)
        
        if TORCHMETRICS_AVAILABLE:
            # torchmetrics를 사용한 효율적인 계산
            # 예측과 동일한 디바이스로 메트릭 이동
            if seg_pred.device != next(self.parameters()).device:
                seg_pred = seg_pred.to(next(self.parameters()).device)
                seg_target = seg_target.to(next(self.parameters()).device)
                depth_pred = depth_pred.to(next(self.parameters()).device)
                depth_target = depth_target.to(next(self.parameters()).device)
            
            # torchmetrics를 사용한 IoU 계산
            miou = self.train_iou(seg_pred, seg_target) if self.training else self.val_iou(seg_pred, seg_target)
            
            # 깊이에 대한 MSE 계산
            mse = self.train_mse(depth_pred, depth_target) if self.training else self.val_mse(depth_pred, depth_target)
            
            metrics = {
                "miou": miou,
                "seg_acc": seg_acc,
                "mse": mse,
                "abs_rel": depth_metrics["abs_rel"],
                "sq_rel": depth_metrics["sq_rel"],
                "rmse": depth_metrics["rmse"],
                "rmse_log": depth_metrics["rmse_log"],
                "delta1": depth_metrics["delta1"],
                "delta2": depth_metrics["delta2"],
                "delta3": depth_metrics["delta3"],
            }
        else:
            # 수동 계산으로 폴백
            # 클래스별 IoU 계산
            tp = torch.zeros(self.num_classes, device=seg_pred.device)
            fp = torch.zeros(self.num_classes, device=seg_pred.device)
            fn = torch.zeros(self.num_classes, device=seg_pred.device)
            
            for c in range(self.num_classes):
                tp[c] = ((seg_pred == c) & (seg_target == c)).float().sum()
                fp[c] = ((seg_pred == c) & (seg_target != c)).float().sum()
                fn[c] = ((seg_pred != c) & (seg_target == c)).float().sum()
            
            # IoU 계산
            iou = tp / (tp + fp + fn + 1e-8)
            miou = iou.mean()
            
            metrics = {
                "miou": miou,
                "seg_acc": seg_acc,
                "abs_rel": depth_metrics["abs_rel"],
                "sq_rel": depth_metrics["sq_rel"],
                "rmse": depth_metrics["rmse"],
                "rmse_log": depth_metrics["rmse_log"],
                "delta1": depth_metrics["delta1"],
                "delta2": depth_metrics["delta2"],
                "delta3": depth_metrics["delta3"],
            }
        
        return metrics

    def training_step(self, batch, batch_idx):
        """
        학습 스텝: 순전파, 손실 계산, 메트릭 계산, 로깅
        
        Args:
            batch: 배치 데이터 (rgb, depth, seg_masks, depth_target, filenames)
            batch_idx: 배치 인덱스
            
        Returns:
            torch.Tensor: 총 손실 값
        """
        rgb, depth, seg_masks, depth_target = batch[:4]
        seg_logits, depth_pred = self(rgb, depth)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # 메트릭 로깅
        self.log("train_loss", total_loss, prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_seg_loss", loss_dict["seg"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        if self._use_dice:
            self.log("train_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("train_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("train_depth_loss", loss_dict["depth"], prog_bar=False, sync_dist=True)
        self.log("train_weighted_seg", loss_dict["weighted_seg"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_weighted_depth", loss_dict["weighted_depth"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_rmse", metrics["rmse"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        # train_miou는 제거 - train_epoch_iou가 정확한 값
        self.log("train_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        
        
        # 불확실성 가중치 로깅
        if self.use_uncertainty_weighting:
            self.log("log_var_seg", loss_dict["log_var_seg"], prog_bar=False, sync_dist=True)
            self.log("log_var_depth", loss_dict["log_var_depth"], prog_bar=False, sync_dist=True)
            self.log("train_weighted_seg", loss_dict["weighted_seg"], prog_bar=True, sync_dist=True)
            self.log("train_weighted_depth", loss_dict["weighted_depth"], prog_bar=True, sync_dist=True)
        
        # 에폭 레벨 IoU 계산을 위한 클래스별 통계 누적
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        
        # 첫 번째 스텝에서 올바른 디바이스에 누적기 초기화
        if self._epoch_train_tp is None:
            device = seg_pred.device
            self._epoch_train_tp = torch.zeros(self.num_classes, device=device)
            self._epoch_train_fp = torch.zeros(self.num_classes, device=device)
            self._epoch_train_fn = torch.zeros(self.num_classes, device=device)
        
        # 클래스별 True Positive, False Positive, False Negative 계산
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            self._epoch_train_tp[c] += tp
            self._epoch_train_fp[c] += fp
            self._epoch_train_fn[c] += fn

        # 첫 번째 스텝에서 메트릭 누적기 초기화
        if self._train_metrics_accumulator is None:
            self._train_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        # 메트릭을 효율적으로 저장 (GPU에서 유지)
        metrics_to_store = {
            "loss": total_loss,
            "seg_loss": loss_dict["seg"],
            "seg_ce": loss_dict["seg_ce"],
            "depth_loss": loss_dict["depth"],
            "miou": metrics["miou"],
            "abs_rel": metrics["abs_rel"],
            "sq_rel": metrics["sq_rel"],
            "rmse": metrics["rmse"],
            "rmse_log": metrics["rmse_log"],
        }
        
        # Dice 손실이 사용되는 경우 추가
        if self._use_dice and "seg_dice" in loss_dict:
            metrics_to_store["seg_dice"] = loss_dict["seg_dice"]
        
        self._train_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb, depth, seg_masks, depth_target = batch[:4]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg_logits, depth_pred = self(rgb, depth)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # FPS - 단일 이미지 기준으로 계산
        per_image_time = dt / rgb.shape[0]
        fps = 1.0 / per_image_time
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        if self._use_dice:
            self.log("val_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("val_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("val_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True)
        self.log("val_weighted_seg", loss_dict["weighted_seg"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_weighted_depth", loss_dict["weighted_depth"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        # val_miou는 제거 - val_epoch_iou가 정확한 값
        self.log("val_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_rmse", metrics["rmse"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_rmse_log", metrics["rmse_log"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_delta1", metrics["delta1"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_delta2", metrics["delta2"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_delta3", metrics["delta3"], prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_fps", fps, prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        
        if self.use_uncertainty_weighting:
            self.log("val_log_var_seg", loss_dict["log_var_seg"], prog_bar=False, sync_dist=True)
            self.log("val_log_var_depth", loss_dict["log_var_depth"], prog_bar=False, sync_dist=True)
        
        # torchmetrics를 사용한 IoU 업데이트 (누적만, 로깅 안함)
        if TORCHMETRICS_AVAILABLE:
            seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
            self.val_iou(seg_pred, seg_masks)
        
        # No validation-stage visual saving
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        
        # Initialize accumulators on correct device if first time
        if self._epoch_val_tp is None:
            device = seg_pred.device
            self._epoch_val_tp = torch.zeros(self.num_classes, device=device)
            self._epoch_val_fp = torch.zeros(self.num_classes, device=device)
            self._epoch_val_fn = torch.zeros(self.num_classes, device=device)
        
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            self._epoch_val_tp[c] += tp
            self._epoch_val_fp[c] += fp
            self._epoch_val_fn[c] += fn

        # Combined metric for balanced early stopping
        if not self.first_val_done:
            self.seg_best = metrics["miou"].clone()
            self.depth_best = metrics["abs_rel"].clone()
            self.first_val_done = True
        
        self.seg_best = torch.maximum(self.seg_best, metrics["miou"])
        self.depth_best = torch.minimum(self.depth_best, metrics["abs_rel"])
        
        seg_normalized = 1.0 - (metrics["miou"] / (self.seg_best + 1e-6))
        depth_normalized = metrics["abs_rel"] / (self.depth_best + 1e-6)
        combined_metric = 0.5 * seg_normalized + 0.5 * depth_normalized
        
        self.log("val_combined_metric", combined_metric, prog_bar=True, sync_dist=True)
        
        # Store for epoch summary logging
        self.logged_metrics["val_loss"] = total_loss
        self.logged_metrics["val_abs_rel"] = metrics["abs_rel"]
        
        # 첫 번째 스텝에서 메트릭 누적기 초기화
        if self._val_metrics_accumulator is None:
            self._val_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        # 메트릭을 효율적으로 저장 (GPU에서 유지)
        metrics_to_store = {
            "loss": total_loss,
            "seg_loss": loss_dict["seg"],
            "seg_ce": loss_dict["seg_ce"],
            "depth_loss": loss_dict["depth"],
            "miou": metrics["miou"],
            "abs_rel": metrics["abs_rel"],
            "sq_rel": metrics["sq_rel"],
            "rmse": metrics["rmse"],
            "rmse_log": metrics["rmse_log"],
            "delta1": metrics["delta1"],
            "delta2": metrics["delta2"],
            "delta3": metrics["delta3"],
        }
        
        # Dice 손실이 사용되는 경우 추가
        if self._use_dice and "seg_dice" in loss_dict:
            metrics_to_store["seg_dice"] = loss_dict["seg_dice"]
        
        self._val_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        rgb, depth, seg_masks, depth_target = batch[:4]
        filenames = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            filenames = batch[4]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg_logits, depth_pred = self(rgb, depth)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # FPS - 단일 이미지 기준으로 계산
        per_image_time = dt / rgb.shape[0]
        fps = 1.0 / per_image_time
        
        # Logging
        self.log("test_loss", total_loss, sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_seg_loss", loss_dict["seg"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_depth_loss", loss_dict["depth"], sync_dist=True, batch_size=rgb.shape[0])
        # test_miou는 제거 - test_epoch_iou가 정확한 값
        self.log("test_abs_rel", metrics["abs_rel"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_sq_rel", metrics["sq_rel"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_rmse", metrics["rmse"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_rmse_log", metrics["rmse_log"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_delta1", metrics["delta1"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_delta2", metrics["delta2"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_delta3", metrics["delta3"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_acc", metrics["seg_acc"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_fps", fps, sync_dist=True, batch_size=rgb.shape[0])
        
        if self.use_uncertainty_weighting:
            self.log("test_log_var_seg", loss_dict["log_var_seg"], sync_dist=True)
            self.log("test_log_var_depth", loss_dict["log_var_depth"], sync_dist=True)
        
        # torchmetrics를 사용한 IoU 업데이트 (누적만, 로깅 안함)
        if TORCHMETRICS_AVAILABLE:
            seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
            self.test_iou(seg_pred, seg_masks)
        
        # 클래스별 IoU 집계를 위한 통계 누적 (테스트 에폭 레벨)
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        if not hasattr(self, "_epoch_test_tp") or self._epoch_test_tp is None:
            device = seg_pred.device
            self._epoch_test_tp = torch.zeros(self.num_classes, device=device)
            self._epoch_test_fp = torch.zeros(self.num_classes, device=device)
            self._epoch_test_fn = torch.zeros(self.num_classes, device=device)
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            self._epoch_test_tp[c] += tp
            self._epoch_test_fp[c] += fp
            self._epoch_test_fn[c] += fn
        
        # Save visuals for test: save for ALL batches (full dataset inference)
        self._maybe_save_visuals(rgb, depth, seg_masks, depth_target, seg_logits, depth_pred,
                                stage="test", batch_idx=batch_idx, filenames=filenames)
        
        return total_loss

    def on_test_epoch_end(self) -> None:
        """테스트 에폭 종료 시 정확한 에폭별 mIoU 및 클래스별 IoU 계산"""
        # torchmetrics를 사용한 정확한 mIoU 계산
        if TORCHMETRICS_AVAILABLE:
            epoch_iou = self.test_iou.compute()
            self.log("test_miou", epoch_iou, prog_bar=True, sync_dist=True)
            self.test_iou.reset()
        
        # 클래스별 IoU 계산 및 로깅
        if hasattr(self, "_epoch_test_tp") and self._epoch_test_tp is not None:
            self._compute_class_iou(self._epoch_test_tp, self._epoch_test_fp, self._epoch_test_fn, "test")
            
            # Reset accumulators
            self._epoch_test_tp = None
            self._epoch_test_fp = None
            self._epoch_test_fn = None

    def on_train_epoch_end(self) -> None:
        # Use memory-efficient metrics accumulator
        if self._train_metrics_accumulator is not None:
            epoch_metrics = self._train_metrics_accumulator.compute_mean()
            
            # Update curves
            for key, value in epoch_metrics.items():
                curve_key = f"train_{key}"
                if curve_key in self.curves:
                    self.curves[curve_key].append(value)
            
            # Reset accumulator for next epoch
            self._train_metrics_accumulator.reset()
        
        # Log epoch-level metrics using torchmetrics
        if TORCHMETRICS_AVAILABLE:
            # Compute epoch-level IoU (정확한 에폭 평균)
            epoch_iou = self.train_iou.compute()
            self.log("train_miou", epoch_iou, prog_bar=True, sync_dist=True)
            self.train_iou.reset()
            
            # Compute epoch-level MSE
            epoch_mse = self.train_mse.compute()
            self.log("train_epoch_mse", epoch_mse, prog_bar=False, sync_dist=True)
            self.train_mse.reset()
        else:
            # Fallback to manual computation
            if self._epoch_train_tp is not None:
                class_names = [
                    "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
                ]
                for i, class_name in enumerate(class_names[: self.num_classes]):
                    denom = (self._epoch_train_tp[i] + self._epoch_train_fp[i] + self._epoch_train_fn[i])
                    iou_value = (self._epoch_train_tp[i] / denom) if denom > 0 else torch.tensor(0.0)
                    self.log(f"train_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)

        # Reset accumulators
        if self._epoch_train_tp is not None:
            self._epoch_train_tp.zero_()
            self._epoch_train_fp.zero_()
            self._epoch_train_fn.zero_()

    def on_validation_epoch_end(self) -> None:
        # Use memory-efficient metrics accumulator
        if self._val_metrics_accumulator is not None:
            epoch_metrics = self._val_metrics_accumulator.compute_mean()
            
            # Update curves
            for key, value in epoch_metrics.items():
                curve_key = f"val_{key}"
                if curve_key in self.curves:
                    self.curves[curve_key].append(value)
            
            # Reset accumulator for next epoch
            self._val_metrics_accumulator.reset()
        
        # Log epoch-level metrics using torchmetrics
        if TORCHMETRICS_AVAILABLE:
            # Compute epoch-level IoU (정확한 에폭 평균)
            epoch_iou = self.val_iou.compute()
            self.log("val_miou", epoch_iou, prog_bar=True, sync_dist=True)
            # cache for summary print
            try:
                self._last_val_epoch_iou = float(epoch_iou.detach().cpu().item())
            except Exception:
                self._last_val_epoch_iou = None
            self.val_iou.reset()
            
            # Compute epoch-level MSE
            epoch_mse = self.val_mse.compute()
            self.log("val_epoch_mse", epoch_mse, prog_bar=False, sync_dist=True)
            self.val_mse.reset()
        
        # 클래스별 IoU 계산 및 로깅 (torchmetrics 사용 여부와 관계없이)
        if self._epoch_val_tp is not None:
            self._compute_class_iou(self._epoch_val_tp, self._epoch_val_fp, self._epoch_val_fn, "val")

        
        # 로그 출력은 CustomEarlyStopping 콜백에서 처리됩니다

        # Reset accumulators
        if self._epoch_val_tp is not None:
            self._epoch_val_tp.zero_()
            self._epoch_val_fp.zero_()
            self._epoch_val_fn.zero_()

    def on_fit_end(self) -> None:
        """Save final visualizations and curves at the end of training"""
        try:
            # Final-stage visual saving disabled (keep only test-stage visuals)
            
            # Save curves
            if not self.save_root_dir:
                return
            curves_dir = os.path.join(self.save_root_dir, "curves")
            os.makedirs(curves_dir, exist_ok=True)

            # Loss curves
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Total loss
            ax = axes[0, 0]
            if len(self.curves["train_loss"]) > 0:
                ax.plot(self.curves["train_loss"], label="train")
            if len(self.curves["val_loss"]) > 0:
                ax.plot(self.curves["val_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Total Loss")
            ax.grid(True)
            ax.legend()
            
            # Segmentation loss
            ax = axes[0, 1]
            if len(self.curves["train_seg_loss"]) > 0:
                ax.plot(self.curves["train_seg_loss"], label="train")
            if len(self.curves["val_seg_loss"]) > 0:
                ax.plot(self.curves["val_seg_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Segmentation Loss")
            ax.grid(True)
            ax.legend()
            
            # Depth loss
            ax = axes[1, 0]
            if len(self.curves["train_depth_loss"]) > 0:
                ax.plot(self.curves["train_depth_loss"], label="train")
            if len(self.curves["val_depth_loss"]) > 0:
                ax.plot(self.curves["val_depth_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Depth Loss")
            ax.grid(True)
            ax.legend()
            
            # mIoU
            ax = axes[1, 1]
            if len(self.curves["train_miou"]) > 0:
                ax.plot(self.curves["train_miou"], label="train")
            if len(self.curves["val_miou"]) > 0:
                ax.plot(self.curves["val_miou"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("mIoU")
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "loss_curves.png"))
            plt.close()
            
            # Depth metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # AbsRel
            ax = axes[0]
            if len(self.curves["train_abs_rel"]) > 0:
                ax.plot(self.curves["train_abs_rel"], label="train")
            if len(self.curves["val_abs_rel"]) > 0:
                ax.plot(self.curves["val_abs_rel"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("AbsRel (lower is better)")
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend()
            
            # RMSE
            ax = axes[1]
            if len(self.curves["train_rmse"]) > 0:
                ax.plot(self.curves["train_rmse"], label="train")
            if len(self.curves["val_rmse"]) > 0:
                ax.plot(self.curves["val_rmse"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("RMSE (lower is better)")
            ax.grid(True)
            ax.legend()
            
            # δ<1.25
            ax = axes[2]
            if len(self.curves["val_delta1"]) > 0:
                ax.plot(self.curves["val_delta1"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("δ<1.25 (higher is better)")
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "depth_metrics.png"))
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Failed to save curves: {e}")

    @torch.no_grad()
    def _maybe_save_visuals(self, rgb: torch.Tensor, depth: torch.Tensor, seg_masks: torch.Tensor, 
                           depth_target: torch.Tensor, seg_logits: torch.Tensor, 
                           depth_pred: torch.Tensor, stage: str, batch_idx: int,
                           filenames: Optional[List[str]] = None) -> None:
        if not self.base_vis_dir or self.vis_max <= 0:
            return
        # For test stage, allow saving for all batches. For others, only first batch.
        if stage != "test" and batch_idx != 0:
            return
        
        try:
            import os
            from PIL import Image as PILImage
            os.makedirs(os.path.join(self.base_vis_dir, stage), exist_ok=True)

            # Denormalize RGB images (keep on GPU for torchvision processing)
            imgs = (rgb * self.vis_std + self.vis_mean).clamp(0, 1)
            imgs = (imgs * 255.0).to(torch.uint8)

            # Segmentation predictions (keep on GPU for torchvision processing)
            seg_preds = torch.softmax(seg_logits, dim=1).argmax(dim=1).to(torch.int64)
            seg_gts = seg_masks.to(torch.int64)
            
            # Depth predictions and targets (keep on GPU for torchvision processing)
            depth_preds = depth_pred.squeeze(1)
            depth_gts = depth_target

            pal = self.palette.cpu()
            
            def colorize_seg(label_hw: torch.Tensor) -> np.ndarray:
                # Ensure CPU numpy array
                label_np = label_hw.detach().cpu().numpy()
                h, w = label_np.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(min(self.num_classes, pal.shape[0])):
                    out[label_np == c] = pal[c].numpy()
                return out
            
            def colorize_depth(depth_hw: torch.Tensor) -> np.ndarray:
                """Colorize depth with viridis colormap (pure numpy implementation)"""
                depth_np = depth_hw.detach().cpu().numpy()
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                
                # Pure numpy viridis colormap implementation
                # Viridis colormap: blue -> green -> yellow
                depth_colored = np.zeros((*depth_norm.shape, 3), dtype=np.uint8)
                
                # Blue to green transition (0.0 to 0.5)
                mask1 = depth_norm <= 0.5
                if mask1.any():
                    t = depth_norm[mask1] * 2.0  # 0 to 1
                    depth_colored[mask1, 0] = (68 * (1 - t) + 34 * t).astype(np.uint8)  # Blue to green
                    depth_colored[mask1, 1] = (1 * (1 - t) + 139 * t).astype(np.uint8)  # Blue to green
                    depth_colored[mask1, 2] = (84 * (1 - t) + 34 * t).astype(np.uint8)   # Blue to green
                
                # Green to yellow transition (0.5 to 1.0)
                mask2 = depth_norm > 0.5
                if mask2.any():
                    t = (depth_norm[mask2] - 0.5) * 2.0  # 0 to 1
                    depth_colored[mask2, 0] = (34 * (1 - t) + 253 * t).astype(np.uint8)  # Green to yellow
                    depth_colored[mask2, 1] = (139 * (1 - t) + 231 * t).astype(np.uint8) # Green to yellow
                    depth_colored[mask2, 2] = (34 * (1 - t) + 37 * t).astype(np.uint8)   # Green to yellow
                
                return depth_colored

            # For test stage, save all images in the batch; otherwise respect vis_max
            save_count = imgs.shape[0] if stage == "test" else min(self.vis_max, imgs.shape[0])
            
            # Use unified PIL-based visualization
            self._save_visuals_pil(imgs, seg_gts, seg_preds, depth_gts, depth_preds, 
                                 stage, save_count, filenames, colorize_seg, colorize_depth)
        except Exception as e:
            warnings.warn(f"Failed to save visualization: {e}")
    
    def _save_visuals_pil(self, imgs, seg_gts, seg_preds, depth_gts, depth_preds, 
                         stage, save_count, filenames, colorize_seg, colorize_depth):
        """Fallback PIL-based visualization"""
        try:
            for i in range(save_count):
                # Move to CPU before numpy conversion
                img = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
                
                # Segmentation results
                seg_gt = colorize_seg(seg_gts[i])
                seg_pr = colorize_seg(seg_preds[i])
                
                # Depth results
                depth_gt = colorize_depth(depth_gts[i])
                depth_pr = colorize_depth(depth_preds[i])
                
                # Resize depth to match image size if needed
                if depth_gt.shape[:2] != img.shape[:2]:
                    depth_gt = np.array(Image.fromarray(depth_gt).resize((img.shape[1], img.shape[0])))
                    depth_pr = np.array(Image.fromarray(depth_pr).resize((img.shape[1], img.shape[0])))
                
                # Create standardized 2x3 panel
                row1 = np.concatenate([seg_gt, img, seg_pr], axis=1)
                row2 = np.concatenate([depth_gt, img, depth_pr], axis=1)
                panel = np.concatenate([row1, row2], axis=0)
                
                if filenames is not None and i < len(filenames):
                    stem = os.path.splitext(os.path.basename(filenames[i]))[0]
                    out_filename = f"{stem}.png"
                else:
                    out_filename = f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{i}.png"
                out_path = os.path.join(self.base_vis_dir, stage, out_filename)
                # 컨텍스트 매니저 사용으로 메모리 누수 방지
                with Image.fromarray(panel) as img:
                    img.save(out_path)
                
        except Exception as e:
            warnings.warn(f"Failed to save PIL visualization: {e}")
    
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.t_max), eta_min=self.final_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# Dataset Builder
# ============================================================================
def build_esanet_datasets(dataset_root: Path, image_size: Tuple[int, int]):
    """
    ESANet 멀티태스크 학습을 위한 데이터셋을 구축합니다.
    
    Args:
        dataset_root: 데이터셋 루트 디렉토리 경로
        image_size: 모델 입력 이미지 크기 (width, height)
        
    Returns:
        tuple: (train_ds, val_ds, test_ds) 학습/검증/테스트 데이터셋
        
    데이터셋 구조:
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
    """
    # 데이터셋 디렉토리 경로 설정
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "masks"
    train_depth = dataset_root / "train" / "depth"
    
    val_images = dataset_root / "val" / "images"
    val_masks = dataset_root / "val" / "masks"
    val_depth = dataset_root / "val" / "depth"
    
    test_images = dataset_root / "test" / "images"
    test_masks = dataset_root / "test" / "masks"
    test_depth = dataset_root / "test" / "depth"

    # ImageNet 표준 정규화 파라미터 가져오기
    mean, std = get_preprocessing_params("esanet")

    # 학습 데이터셋: 전처리 파이프라인 일관성 유지
    train_ds = RGBDepthMultiTaskDataset(
        train_images, train_masks, train_depth,
        image_size=image_size, mean=mean, std=std, is_train=True
    )
    
    # 검증 데이터셋: 전처리 파이프라인 일관성 유지
    val_ds = RGBDepthMultiTaskDataset(
        val_images, val_masks, val_depth,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    
    # 테스트 데이터셋: 전처리 파이프라인 일관성 유지
    test_ds = RGBDepthMultiTaskDataset(
        test_images, test_masks, test_depth,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Collate Function (avoids non-resizable storage issues)
# ============================================================================
def collate_esanet_batch(batch):
    """
    ESANet 배치 데이터를 안전하게 스택하는 커스텀 콜레이트 함수입니다.
    
    Args:
        batch: 샘플 리스트, 각 샘플은 (image, depth, mask, depth_target, filename) 튜플
        
    Returns:
        tuple: 스택된 배치 텐서들
            - imgs: RGB 이미지 배치 [B, 3, H, W]
            - depths: 깊이 입력 배치 [B, 1, H, W]  
            - masks: 세그멘테이션 마스크 배치 [B, H, W]
            - depth_targets: 깊이 타겟 배치 [B, H, W]
            - filenames: 파일명 리스트 (선택사항)
    
    주요 기능:
    - 메모리 리사이즈 오류 방지: contiguous() 호출로 독립적 스토리지 보장
    - 배치 크기 동적 처리: 다양한 배치 크기에 대응
    - 파일명 처리: 테스트 시 파일명 정보 보존
    """
    # 배치 구성 요소별로 분리하여 리스트에 저장
    imgs = []
    depths = []
    masks = []
    depth_targets = []
    filenames = []
    
    # 각 샘플을 순회하며 텐서를 독립적으로 처리
    for sample in batch:
        if len(sample) >= 4:
            img, depth, mask, depth_t = sample[:4]
            # contiguous() 호출로 독립적 스토리지 보장 (메모리 오류 방지)
            imgs.append(img.contiguous())
            depths.append(depth.contiguous())
            masks.append(mask.contiguous())
            depth_targets.append(depth_t.contiguous())
            
            # 파일명이 있는 경우 추가
            if len(sample) >= 5:
                filenames.append(sample[4])
        else:
            raise RuntimeError("Unexpected batch item length: expected >=4")
    
    # 리스트를 배치 텐서로 스택
    imgs = torch.stack(imgs, dim=0)
    depths = torch.stack(depths, dim=0)
    masks = torch.stack(masks, dim=0)
    depth_targets = torch.stack(depth_targets, dim=0)
    
    # 파일명이 있으면 함께 반환, 없으면 텐서만 반환
    if filenames:
        return imgs, depths, masks, depth_targets, filenames
    return imgs, depths, masks, depth_targets

# ============================================================================
# Main
# ============================================================================
def parse_args():
    """
    명령행 인수를 파싱하여 설정 파일 경로와 옵션을 반환합니다.
    
    Returns:
        argparse.Namespace: 파싱된 명령행 인수
            - config: 설정 파일 경로 (YAML 또는 JSON)
            - create_default_config: 기본 설정 파일 생성 여부
    """
    parser = argparse.ArgumentParser(description="Train ESANet Multi-Task (Seg + Depth)")
    # 사용자가 명시적으로 설정 파일을 지정하도록 강제
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml or config.json")
    parser.add_argument("--create-default-config", action="store_true", help="Create default config file and exit")
    return parser.parse_args()


def main() -> None:
    """
    ESANet 멀티태스크 학습의 메인 함수입니다.
    
    주요 기능:
    - 설정 파일 로드 및 검증
    - 데이터셋 구축 및 DataLoader 생성
    - 모델 초기화 및 사전훈련 가중치 로드
    - PyTorch Lightning Trainer 설정 및 학습 실행
    - 검증 및 테스트 수행
    - 학습 결과 요약 및 시각화 저장
    
    처리 순서:
    1) 전역 시드 설정 및 재현성 보장
    2) 설정 파일 로드 (YAML 우선, JSON 폴백)
    3) 데이터셋 경로 검증 및 구축
    4) 모델 및 옵티마이저 초기화
    5) 콜백 및 로거 설정
    6) 학습/검증/테스트 실행
    7) 결과 요약 및 저장
    """
    if not ESANET_AVAILABLE:
        print("❌ ESANet 모델을 사용할 수 없습니다.")
        return
    
    # 전역 결정론 설정: 모든 랜덤 시드를 고정하여 재현성 보장
    set_global_determinism(42)
    pl.seed_everything(42, workers=True)
    
    # Tensor Core 최적화 (RTX 4090용)
    # 16비트 혼합 정밀도 학습 시 성능 향상을 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    args = parse_args()
    
    # 기본 설정 파일 생성 요청 처리
    if args.create_default_config:
        from config import create_default_config_file
        create_default_config_file("default_config.yaml")
        print("✅ Default configuration file created: default_config.yaml")
        return
    
    # 설정 파일 로드 (외부 모듈 없이 YAML 직접 로드)
    def _dict_to_namespace(obj):
        from types import SimpleNamespace
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [ _dict_to_namespace(v) for v in obj ]
        return obj

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.endswith('.yaml') or cfg_path.endswith('.yml'):
        try:
            import yaml
        except Exception as e:
            raise ImportError(f"YAML 로더(pyyaml)가 필요합니다: {e}")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f) or {}
        config = _dict_to_namespace(cfg_dict)
        print(f"✅ Configuration loaded from {cfg_path}")
    else:
        # JSON 폴백 - 주석 손실 경고
        warnings.warn("JSON config는 주석을 지원하지 않습니다. YAML 사용을 권장합니다.")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_dict = json.load(f)
        config = _dict_to_namespace(cfg_dict)
        print(f"✅ Configuration (JSON) loaded from {cfg_path}")

    # 설정 정보 디버그 출력
    print(f"📋 Config type: {type(config)}")
    print(f"📋 Dataset root: {getattr(getattr(config, 'data', object()), 'dataset_root', 'N/A')}")
    
    # CUDA 환경 정보 출력
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # 설정 기반 경로 설정
    dataset_root = Path(os.path.abspath(config.data.dataset_root))
    output_dir = Path(os.path.abspath(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 실행 시간 측정 (학습 + 검증 + 테스트)
    _run_start_time = time.time()

    # 데이터셋 구축 (전처리 파이프라인 일관성 유지)
    train_ds, val_ds, test_ds = build_esanet_datasets(
        dataset_root, (config.model.width, config.model.height)
    )
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 학습용 DataLoader: 셔플 활성화
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_batch,
    )
    
    # 검증용 DataLoader: 셔플 비활성화
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_batch,
    )
    
    # 테스트용 DataLoader: 셔플 비활성화
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_batch,
    )

    steps_per_epoch = max(1, len(train_loader))
    
    # 시각화 디렉토리 경로 해결
    vis_dir = config.visualization.vis_dir
    if vis_dir.strip().lower() == "none":
        vis_dir = ""
    elif vis_dir.strip() == "":
        vis_dir = str(output_dir / "vis")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # 사전훈련 가중치 경로 해결: 설정 우선, 없으면 기본 NYUv2 가중치 사용
    use_pretrained = getattr(config.model, 'use_pretrained', True)
    pretrained_path_cfg = getattr(config.model, 'pretrained_path', None)
    default_pretrained_path = \
        "/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth"
    pretrained_path_to_use = None
    if use_pretrained:
        pretrained_path_to_use = pretrained_path_cfg if pretrained_path_cfg else (
            default_pretrained_path if os.path.exists(default_pretrained_path) else None
        )
    if use_pretrained and pretrained_path_to_use:
        print(f"✅ Using ESANet pretrained weights: {pretrained_path_to_use}")
    elif use_pretrained:
        print("📝 No ESANet pretrained weights found; training from scratch...")
    else:
        print("ℹ️ Config: use_pretrained=False, training from scratch.")

    # ESANet 멀티태스크 모델 초기화
    model = LightningESANetMTL(
        height=config.model.height,
        width=config.model.width,
        num_classes=config.model.num_classes,
        encoder_rgb=config.model.encoder_rgb,
        encoder_depth=config.model.encoder_depth,
        encoder_block=config.model.encoder_block,
        lr=config.training.lr,
        scheduler_t_max=getattr(config.training, 'scheduler_t_max', config.training.epochs * steps_per_epoch),
        final_lr=getattr(config.training, 'final_lr', 1e-5),
        loss_type=config.training.loss_type,
        seg_loss_weight=config.training.seg_loss_weight,
        depth_loss_weight=config.training.depth_loss_weight,
        use_uncertainty_weighting=config.training.use_uncertainty_weighting,
        use_dwa_weighting=config.training.use_dwa_weighting,
        save_vis_dir=vis_dir,
        vis_max=config.visualization.vis_max,
        save_root_dir=str(output_dir),
        pretrained_path=pretrained_path_to_use,
    )

    # 학습 콜백 설정
    # mIoU 기반 체크포인트: 세그멘테이션 성능이 가장 좋은 모델 저장
    ckpt_miou = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="esanet-mtl-miou-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,  # 최고 mIoU만 저장
        save_last=False,  # 마지막 체크포인트 저장 비활성화 (last.ckpt 방지)
    )
    
    # AbsRel 기반 체크포인트: 깊이 추정 성능이 가장 좋은 모델 저장
    ckpt_absrel = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="esanet-mtl-absrel-{epoch:02d}-{val_abs_rel:.4f}",
        monitor="val_abs_rel",
        mode="min",
        save_top_k=1,  # 최소 AbsRel만 저장
    )

    # 조기 종료: 깊이 추정 성능이 개선되지 않으면 학습 중단
    # 설정 기반 조기 종료 설정
    es_monitor = getattr(config.training, 'early_stop_monitor', 'val_abs_rel')
    # YAML에서 반드시 가져오도록 하드코딩 기본값 제거
    es_patience = config.training.early_stop_patience
    es_min_delta = config.training.early_stop_min_delta
    # 모니터명에 따라 모드 자동 선정
    es_mode = 'min'
    try:
        mon_lower = str(es_monitor).lower()
        if 'iou' in mon_lower or 'acc' in mon_lower:
            es_mode = 'max'
    except Exception:
        pass
    
    

    early_stop = EarlyStopping(
        monitor=es_monitor,
        min_delta=es_min_delta,
        patience=es_patience,
        verbose=True,  # PyTorch Lightning이 직접 로그 출력
        mode=es_mode,
    )

    # TensorBoard 로거: output_dir에 직접 로그 저장 (version_x 중첩 방지)
    logger = TensorBoardLogger(save_dir=str(output_dir), name="", version="")

    # PyTorch Lightning Trainer 설정
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        precision=config.system.precision,
        default_root_dir=str(output_dir),
        callbacks=[ckpt_miou, ckpt_absrel, early_stop],
        logger=logger,
        accelerator=config.system.accelerator,
        devices=config.system.devices,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    # 학습 시작
    trainer.fit(model, train_loader, val_loader, ckpt_path=(config.system.ckpt_path or None))
    
    # 최고 성능 체크포인트로 검증 수행
    print("=" * 80)
    print("Validating with best checkpoint (by lowest val_abs_rel)...")
    print("=" * 80)
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt_absrel.best_model_path)
    
    # 최고 성능 체크포인트로 테스트 수행
    print("=" * 80)
    print("Testing with best checkpoint (by lowest val_abs_rel)...")
    print("=" * 80)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_absrel.best_model_path)

    # 총 실행 시간 계산
    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)
    
    # 학습 결과 요약
    # (on_test_epoch_end에서 기록된 클래스별 IoU는 TensorBoard 이벤트에 저장됨)
    summary = {
        "best_checkpoint_miou": ckpt_miou.best_model_path,
        "best_miou": float(ckpt_miou.best_model_score) if ckpt_miou.best_model_score is not None else None,
        "best_checkpoint_absrel": ckpt_absrel.best_model_path,
        "best_absrel": float(ckpt_absrel.best_model_score) if ckpt_absrel.best_model_score is not None else None,
        "best_val_results": best_val_results,
        "test_results": test_results,
        "training_time_sec": round(_elapsed_sec, 2),
        "training_time_hms": f"{_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d}",
    }
    # 클래스별 mIoU를 최종 로그 파일에 함께 저장 (validation과 test 에폭 누적치를 이용)
    try:
        class_names = [
            "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
        ][: model.num_classes]
        
        # Validation 클래스별 IoU 계산 비활성화 (필요시 주석 해제)
        # val_class_ious = {}
        # if hasattr(model, "_epoch_val_tp") and model._epoch_val_tp is not None:
        #     for i, class_name in enumerate(class_names):
        #         denom = (model._epoch_val_tp[i] + model._epoch_val_fp[i] + model._epoch_val_fn[i])
        #         if float(denom.detach().cpu().item()) > 0.0:
        #             iou_value = (model._epoch_val_tp[i] / denom).detach().cpu().item()
        #         else:
        #             iou_value = 0.0
        #         val_class_ious[class_name] = iou_value
        # summary["val_class_iou"] = val_class_ious
        
        # Test 클래스별 IoU 계산 비활성화 (필요시 주석 해제)
        # test_class_ious = {}
        # if hasattr(model, "_epoch_test_tp") and model._epoch_test_tp is not None:
        #     for i, class_name in enumerate(class_names):
        #         denom = (model._epoch_test_tp[i] + model._epoch_test_fp[i] + model._epoch_test_fn[i])
        #         if float(denom.detach().cpu().item()) > 0.0:
        #             iou_value = (model._epoch_test_tp[i] / denom).detach().cpu().item()
        #         else:
        #             iou_value = 0.0
        #         test_class_ious[class_name] = iou_value
        # summary["test_class_iou"] = test_class_ious
    except Exception:
        pass

    print("=" * 80)
    print("학습완료!")
    print("=" * 80)
    print(summary)
    print(f"⏱️ 몇분이나 걸렸을까요?: {_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d} ({_elapsed_sec:.2f}s)")

    # 결과를 CSV로도 저장 (요약 및 테스트/검증 주요 지표)
    try:
        csv_file = output_dir / "results.csv"
        fieldnames = [
            "training_time_sec", "training_time_hms",
            "best_miou", "best_absrel",
        ]
        # 검증/테스트 주요 지표를 평탄화해서 추가
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            for k in best_val_results[0].keys():
                if k not in fieldnames:
                    fieldnames.append(f"val::{k}")
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            for k in test_results[0].keys():
                if k not in fieldnames:
                    fieldnames.append(f"test::{k}")
        # 클래스별 IoU는 제거됨

        # 한 줄 기록용 데이터 구성
        row = {
            "training_time_sec": summary.get("training_time_sec", None),
            "training_time_hms": summary.get("training_time_hms", None),
            "best_miou": summary.get("best_miou", None),
            "best_absrel": summary.get("best_absrel", None),
        }
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            for k, v in best_val_results[0].items():
                row[f"val::{k}"] = v
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            for k, v in test_results[0].items():
                row[f"test::{k}"] = v
        # 클래스별 IoU는 제거됨

        # CSV 작성 (헤더 포함, 기존 파일 있으면 덮어씀)
        with open(csv_file, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"🧾 CSV 결과가 저장되었습니다: {csv_file}")
    except Exception as e:
        warnings.warn(f"Failed to save CSV: {e}")

    # 사람이 읽기 좋은 training.log 저장
    try:
        log_file = output_dir / "training.log"
        lines = []
        lines.append("=" * 80)
        lines.append("ESANet Multi-Task Learning Results")
        lines.append("=" * 80)
        lines.append(f"Training Time: {summary.get('training_time_hms', '')} ({summary.get('training_time_sec', 0)}s)")
        if summary.get("best_miou") is not None:
            try:
                lines.append(f"Best mIoU: {float(summary['best_miou']):.4f}")
            except Exception:
                lines.append(f"Best mIoU: {summary['best_miou']}")
        if summary.get("best_absrel") is not None:
            try:
                lines.append(f"Best AbsRel: {float(summary['best_absrel']):.4f}")
            except Exception:
                lines.append(f"Best AbsRel: {summary['best_absrel']}")
        if summary.get("best_checkpoint_miou"):
            lines.append(f"Best mIoU Checkpoint: {summary['best_checkpoint_miou']}")
        if summary.get("best_checkpoint_absrel"):
            lines.append(f"Best AbsRel Checkpoint: {summary['best_checkpoint_absrel']}")
        lines.append("")
        # Validation block
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            lines.append("Validation Results:")
            for k, v in best_val_results[0].items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        
       
        
        # Test block
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            lines.append("Test Results:")
            for k, v in test_results[0].items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        
        # Test 클래스별 IoU는 제거됨
        with open(log_file, "w", encoding="utf-8") as flog:
            flog.write("\n".join(lines))
        print(f"🗒️ training.log가 저장되었습니다: {log_file}")
    except Exception as e:
        warnings.warn(f"Failed to save training.log: {e}")


if __name__ == "__main__":
    main()