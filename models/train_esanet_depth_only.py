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

try:
    from torchmetrics import MeanSquaredError, MeanAbsoluteError
    TORCHMETRICS_AVAILABLE = True
    print("✅ torchmetrics를 성공적으로 import했습니다.")
except ImportError:
    print("⚠️ torchmetrics가 설치되지 않았습니다. pip install torchmetrics를 실행하세요.")
    TORCHMETRICS_AVAILABLE = False

# PyTorch Lightning 관련 import
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time

# 학습 곡선 저장을 위한 matplotlib import
import matplotlib.pyplot as plt

# ESANet 모델 import
sys.path.append('/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet')
try:
    from src.models.model_one_modality import ESANetOneModality
    ESANET_AVAILABLE = True
    print("✅ ESANet One Modality 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ ESANet One Modality 모델 import 실패: {e}")
    ESANET_AVAILABLE = False


# ============================================================================
# 상수 정의
# ============================================================================

# ============================================================================
# 재현성 및 결정론 유틸리티
# ============================================================================
def set_global_determinism(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def dataloader_worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    import random
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================================
# RGB-Depth 데이터셋 (RGB 입력으로 Depth 예측)
# ============================================================================
class RGBDepthDataset(Dataset):
    """
    RGB 입력으로 Depth를 예측하는 데이터셋입니다.
    """
    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        image_size: Tuple[int, int],
        is_train: bool = False,
    ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.is_train = is_train

        # RGB 파일 목록
        self.rgb_files: List[str] = [
            f for f in sorted(os.listdir(rgb_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.rgb_files) == 0:
            raise RuntimeError(f"No RGB files in {rgb_dir}")

        # Depth 파일 목록 (RGB와 매칭)
        self.depth_files: List[str] = []
        for rgb_file in self.rgb_files:
            # RGB 파일명에서 확장자 제거하고 depth 파일명 생성
            base_name = os.path.splitext(rgb_file)[0]
            depth_file = f"{base_name}_depth.png"  # 또는 다른 명명 규칙
            
            depth_path = depth_dir / depth_file
            if depth_path.exists():
                self.depth_files.append(depth_file)
            else:
                # 다른 확장자나 명명 규칙 시도
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_depth_file = f"{base_name}{ext}"
                    alt_depth_path = depth_dir / alt_depth_file
                    if alt_depth_path.exists():
                        self.depth_files.append(alt_depth_file)
                        break
                else:
                    print(f"⚠️ Warning: No matching depth file for {rgb_file}")

        if len(self.depth_files) == 0:
            raise RuntimeError(f"No matching depth files found in {depth_dir}")

        print(f"📁 Dataset loaded: {len(self.rgb_files)} RGB files, {len(self.depth_files)} depth files")

    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int):
        rgb_name = self.rgb_files[idx]
        depth_name = self.depth_files[idx]
        
        rgb_path = self.rgb_dir / rgb_name
        depth_path = self.depth_dir / depth_name

        # RGB 이미지 로드
        rgb = Image.open(rgb_path).convert("RGB")
        
        # Depth 이미지 로드
        depth = Image.open(depth_path).convert("L")

        # 모든 이미지를 모델 입력 크기로 리사이즈
        rgb = TF.resize(rgb, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
        depth = TF.resize(depth, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)

        # PIL 이미지를 PyTorch 텐서로 변환
        rgb = TF.to_tensor(rgb).contiguous().clone()  # [3, H, W]
        depth = TF.to_tensor(depth).contiguous()  # [1, H, W]

        # 깊이 정규화 (0-1 범위로) - 이미 contiguous()로 독립적 스토리지 보장됨
        depth = depth.squeeze(0)  # [H, W]로 변환

        # RGB 입력으로 Depth 예측
        return rgb, depth, rgb_name


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================
# 손실 함수들
# ============================================================================
class SILogLoss(nn.Module):
    """
    깊이 추정을 위한 스케일 불변 로그 손실(Scale-Invariant Logarithmic Loss)입니다.
    """
    def __init__(self, lambda_variance: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.lambda_variance = lambda_variance
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # 마스크 적용: 유효한 픽셀만 사용
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # 로그 차이 계산: log(pred) - log(target)
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        
        # SILog 손실 계산: MSE - λ * (평균)²
        mse = (log_diff ** 2).mean()
        mean_log_diff = log_diff.mean()
        loss = mse - self.lambda_variance * (mean_log_diff ** 2)
        
        return loss


class L1DepthLoss(nn.Module):
    """
    유효 마스크 처리가 포함된 깊이 추정용 L1 손실 함수입니다.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
# 깊이 추정 메트릭
# ============================================================================
@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """
    표준 깊이 추정 메트릭을 계산합니다.
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
    sq_rel = (((pred - target) ** 2) / (target + eps)).mean()

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
# ESANet 깊이 추정 전용 모델 (Depth 단일 모달리티)
# ============================================================================
class ESANetDepthOnly(nn.Module):
    """
    ESANet One Modality 구조를 사용한 Depth 전용 모델입니다.
    """
    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        encoder: str = 'resnet34',
        encoder_block: str = 'NonBottleneck1D',
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        if not ESANET_AVAILABLE:
            raise ImportError("ESANet One Modality 모델을 사용할 수 없습니다.")
        
        # ESANet One Modality 모델 초기화 (RGB 입력)
        self.esanet = ESANetOneModality(
            height=height,
            width=width,
            num_classes=1,  # 깊이 추정을 위해 1개 클래스로 초기화
            encoder=encoder,
            encoder_block=encoder_block,
            pretrained_on_imagenet=False,  # ImageNet 사전훈련 비활성화 (경로 문제 해결)
            activation='relu',
            input_channels=3,  # RGB 입력
            encoder_decoder_fusion='add',
            context_module='ppm',
            upsampling='bilinear',
        )
        
        # ESANet 내부의 BatchNorm 설정 변경
        self._fix_batchnorm_for_small_batches()
        
        # 사전 학습된 가중치 로드 (40클래스 → 1클래스 변환 필요)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading pretrained ESANet weights from {pretrained_path}")
            self._load_pretrained_weights_safe(pretrained_path)
        else:
            print("📝 No pretrained weights provided, training from scratch...")
        
        # 깊이 헤드 (1개 클래스 특징에서 깊이 예측)
        self.depth_head = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        print(f"🔧 ESANet RGB-Depth Architecture (One Modality):")
        print(f"  - Input: RGB [B,3,H,W]")
        print(f"  - Output: Depth prediction [B,H,W]")
        print(f"  - Encoder: {encoder} (RGB input)")
        print(f"  - ImageNet pretrained: False (경로 문제로 비활성화)")
    
    def _fix_batchnorm_for_small_batches(self):
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
                    
                    if child.affine:
                        group_norm.weight.data = child.weight.data.clone()
                        group_norm.bias.data = child.bias.data.clone()
                    
                    setattr(module, name, group_norm)
                else:
                    fix_batchnorm_recursive(child)
        
        fix_batchnorm_recursive(self.esanet)
    
    def _load_pretrained_weights_safe(self, pretrained_path: str):
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model_dict = self.esanet.state_dict()
            compatible_dict = {}
            
            compatible_count = 0
            incompatible_count = 0
            
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        compatible_dict[k] = v
                        compatible_count += 1
                    else:
                        incompatible_count += 1
                else:
                    incompatible_count += 1
            
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
        
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        ESANet RGB-Depth 모델의 순전파 연산 (RGB 입력으로 Depth 예측)
        
        Args:
            rgb: RGB 입력 텐서 [B, 3, H, W]
        
        Returns:
            depth_pred: 깊이 예측 [B, H, W] (0-1 범위)
        """
        # RGB는 이미 [B, 3, H, W] 형태
        
        # ESANet One Modality 순전파 (RGB 입력)
        esanet_output = self.esanet(rgb)
        
        # ESANet이 훈련 모드에서 여러 출력을 반환하는 경우 처리
        if isinstance(esanet_output, tuple):
            esanet_features = esanet_output[0]
        else:
            esanet_features = esanet_output
        
        # 깊이 헤드 순전파 (1개 클래스 특징에서)
        depth_raw = self.depth_head(esanet_features)
        
        # 시그모이드 적용하여 깊이를 [0, 1] 범위로 제한
        depth_pred = torch.sigmoid(depth_raw)
        
        # 깊이 예측이 타겟과 동일한 형태 [B, H, W]가 되도록 보장
        if depth_pred.dim() == 4 and depth_pred.size(1) == 1:
            depth_pred = depth_pred.squeeze(1)
        
        return depth_pred


# ============================================================================
# 메모리 효율적인 메트릭 누적기
# ============================================================================
class MetricsAccumulator:
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = {}
        self.count = 0
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.device != self.device:
                    value = value.to(self.device)
            else:
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.count += 1
    
    def compute_mean(self) -> Dict[str, float]:
        result = {}
        for key, values in self.metrics.items():
            if values:
                stacked = torch.stack(values)
                mean_value = stacked.mean()
                result[key] = float(mean_value.cpu().item())
        return result
    
    def reset(self):
        for key, values in self.metrics.items():
            if values:
                del values[:]
        self.metrics.clear()
        self.count = 0


# ============================================================================
# PyTorch Lightning 모듈
# ============================================================================
class LightningESANetDepthOnly(pl.LightningModule):
    """
    ESANet 깊이 추정 전용 학습을 위한 PyTorch Lightning 모듈입니다.
    """
    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        encoder: str = 'resnet34',
        encoder_block: str = 'NonBottleneck1D',
        lr: float = 1e-4,
        scheduler_t_max: int = 1000,
        final_lr: float = 1e-5,
        loss_type: str = "silog",  # "silog" or "l1"
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # ESANet Depth 전용 모델 초기화
        self.model = ESANetDepthOnly(
            height=height,
            width=width,
            encoder=encoder,
            encoder_block=encoder_block,
            pretrained_path=pretrained_path,
        )
        
        # 손실 함수 설정
        if loss_type == "silog":
            self.depth_loss_fn = SILogLoss()
        elif loss_type == "l1":
            self.depth_loss_fn = L1DepthLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # 기본 설정 저장
        self.lr = lr
        self.t_max = scheduler_t_max
        self.final_lr = float(final_lr)
        self.base_vis_dir = save_vis_dir
        self.vis_max = int(vis_max) if vis_max is not None else 0
        self.save_root_dir = save_root_dir
        
        # 학습 곡선 저장용 딕셔너리
        self.curves = {
            "train_loss": [],
            "val_loss": [],
            "train_abs_rel": [],
            "val_abs_rel": [],
        }
        
        # 메모리 효율적인 메트릭 누적기
        self._train_metrics_accumulator = None
        self._val_metrics_accumulator = None
        
        # 효율적인 메트릭 계산을 위한 torchmetrics 설정
        if TORCHMETRICS_AVAILABLE:
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
        
        # 시각화 파라미터 설정 (Depth 전용이므로 RGB 정규화 불필요)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)
    
    def _compute_loss(self, depth_pred: torch.Tensor, depth_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        깊이 추정 손실 계산
        """
        depth_loss = self.depth_loss_fn(depth_pred, depth_target)
        
        loss_dict = {
            "total": depth_loss,
            "depth": depth_loss,
        }
        
        return depth_loss, loss_dict

    @torch.no_grad()
    def _compute_metrics(self, depth_pred: torch.Tensor, depth_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        깊이 추정 메트릭 계산
        """
        # 깊이 메트릭 계산
        depth_metrics = compute_depth_metrics(depth_pred, depth_target)
        
        if TORCHMETRICS_AVAILABLE:
            if depth_pred.device != next(self.parameters()).device:
                depth_pred = depth_pred.to(next(self.parameters()).device)
                depth_target = depth_target.to(next(self.parameters()).device)
            
            # torchmetrics를 사용한 MSE/MAE 계산
            mse = self.train_mse(depth_pred, depth_target) if self.training else self.val_mse(depth_pred, depth_target)
            mae = self.train_mae(depth_pred, depth_target) if self.training else self.val_mae(depth_pred, depth_target)
            
            metrics = {
                "mse": mse,
                "mae": mae,
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
            mse = F.mse_loss(depth_pred, depth_target)
            mae = F.l1_loss(depth_pred, depth_target)
            
            metrics = {
                "mse": mse,
                "mae": mae,
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
        rgb, depth_target = batch[:2]
        depth_pred = self(rgb)
        
        total_loss, loss_dict = self._compute_loss(depth_pred, depth_target)
        metrics = self._compute_metrics(depth_pred.detach(), depth_target)
        
        # 메트릭 로깅
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_mse", metrics["mse"], prog_bar=False, sync_dist=True)
        self.log("train_mae", metrics["mae"], prog_bar=False, sync_dist=True)
        self.log("train_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, batch_size=rgb.shape[0])
        
        if self._train_metrics_accumulator is None:
            self._train_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        metrics_to_store = {
            "loss": total_loss,
            "depth_loss": loss_dict["depth"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "abs_rel": metrics["abs_rel"],
        }
        
        self._train_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb, depth_target = batch[:2]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth_pred = self(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(depth_pred, depth_target)
        metrics = self._compute_metrics(depth_pred.detach(), depth_target)
        
        # FPS - 단일 이미지 기준으로 계산
        per_image_time = dt / rgb.shape[0]
        fps = 1.0 / per_image_time
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_mse", metrics["mse"], prog_bar=False, sync_dist=True)
        self.log("val_mae", metrics["mae"], prog_bar=False, sync_dist=True)
        self.log("val_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_fps", fps, prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        
        if self._val_metrics_accumulator is None:
            self._val_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        metrics_to_store = {
            "loss": total_loss,
            "depth_loss": loss_dict["depth"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "abs_rel": metrics["abs_rel"],
        }
        
        self._val_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        rgb, depth_target = batch[:2]
        filenames = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            filenames = batch[2]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth_pred = self(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(depth_pred, depth_target)
        metrics = self._compute_metrics(depth_pred.detach(), depth_target)
        
        # FPS - 단일 이미지 기준으로 계산
        per_image_time = dt / rgb.shape[0]
        fps = 1.0 / per_image_time
        
        # Logging
        self.log("test_loss", total_loss, sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_depth_loss", loss_dict["depth"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_mse", metrics["mse"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_mae", metrics["mae"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_abs_rel", metrics["abs_rel"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_fps", fps, sync_dist=True, batch_size=rgb.shape[0])
        
        # Save visuals for test
        self._maybe_save_visuals(rgb, depth_target, depth_pred,
                                stage="test", batch_idx=batch_idx, filenames=filenames)
        
        return total_loss

    def on_train_epoch_end(self) -> None:
        if self._train_metrics_accumulator is not None:
            epoch_metrics = self._train_metrics_accumulator.compute_mean()
            
            for key, value in epoch_metrics.items():
                curve_key = f"train_{key}"
                if curve_key in self.curves:
                    self.curves[curve_key].append(value)
            
            self._train_metrics_accumulator.reset()
        
        if TORCHMETRICS_AVAILABLE:
            epoch_mse = self.train_mse.compute()
            self.log("train_epoch_mse", epoch_mse, prog_bar=True, sync_dist=True)
            self.train_mse.reset()
            
            epoch_mae = self.train_mae.compute()
            self.log("train_epoch_mae", epoch_mae, prog_bar=True, sync_dist=True)
            self.train_mae.reset()

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
            # Compute epoch-level MSE
            epoch_mse = self.val_mse.compute()
            self.log("val_epoch_mse", epoch_mse, prog_bar=True, sync_dist=True)
            self.val_mse.reset()
            
            # Compute epoch-level MAE
            epoch_mae = self.val_mae.compute()
            self.log("val_epoch_mae", epoch_mae, prog_bar=True, sync_dist=True)
            self.val_mae.reset()

        
        # 로그 출력은 EarlyStopping 콜백(verbose=True)에서 처리됩니다

    def on_fit_end(self) -> None:
        """Save final visualizations and curves at the end of training"""
        try:
            if not self.save_root_dir:
                return
            curves_dir = os.path.join(self.save_root_dir, "curves")
            os.makedirs(curves_dir, exist_ok=True)

            # Loss curves
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Total loss
            ax = axes[0]
            if len(self.curves["train_loss"]) > 0:
                ax.plot(self.curves["train_loss"], label="train")
            if len(self.curves["val_loss"]) > 0:
                ax.plot(self.curves["val_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            
            # AbsRel
            ax = axes[1]
            if len(self.curves["train_abs_rel"]) > 0:
                ax.plot(self.curves["train_abs_rel"], label="train")
            if len(self.curves["val_abs_rel"]) > 0:
                ax.plot(self.curves["val_abs_rel"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("AbsRel")
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "loss_curves.png"))
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Failed to save curves: {e}")

    @torch.no_grad()
    def _maybe_save_visuals(self, rgb: torch.Tensor, depth_target: torch.Tensor, 
                           depth_pred: torch.Tensor, stage: str, batch_idx: int,
                           filenames: Optional[List[str]] = None) -> None:
        if not self.base_vis_dir or self.vis_max <= 0:
            return
        if stage != "test" and batch_idx != 0:
            return
        
        try:
            import os
            from PIL import Image as PILImage
            os.makedirs(os.path.join(self.base_vis_dir, stage), exist_ok=True)

            # Depth predictions and targets
            depth_preds = depth_pred.detach().cpu()
            depth_gts = depth_target.detach().cpu()
            rgb_inputs = rgb.detach().cpu()
            
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

            def create_rgb_visualization(rgb_hwc: torch.Tensor) -> np.ndarray:
                """Convert RGB tensor to numpy array for visualization"""
                rgb_np = rgb_hwc.detach().cpu().numpy()
                # Convert from [C, H, W] to [H, W, C] and scale to 0-255
                rgb_img = (rgb_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                return rgb_img

            save_count = rgb.shape[0] if stage == "test" else min(self.vis_max, rgb.shape[0])
            
            for i in range(save_count):
                depth_gt = colorize_depth(depth_gts[i])
                rgb_img = create_rgb_visualization(rgb_inputs[i])  # RGB input visualization
                depth_pr = colorize_depth(depth_preds[i])
                
                # Create 1x3 panel (rgb_input, depth_gt, depth_pred)
                panel = np.concatenate([rgb_img, depth_gt, depth_pr], axis=1)
                
                if filenames is not None and i < len(filenames):
                    stem = os.path.splitext(os.path.basename(filenames[i]))[0]
                    out_filename = f"{stem}_depth.png"
                else:
                    out_filename = f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{i}_depth.png"
                out_path = os.path.join(self.base_vis_dir, stage, out_filename)
                # 컨텍스트 매니저 사용으로 메모리 누수 방지
                with Image.fromarray(panel) as img:
                    img.save(out_path)
                
        except Exception as e:
            warnings.warn(f"Failed to save visualization: {e}")

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
def build_esanet_depth_datasets(dataset_root: Path, image_size: Tuple[int, int]):
    """
    ESANet RGB-Depth 학습을 위한 데이터셋을 구축합니다 (RGB 입력으로 Depth 예측).
    """
    train_rgb = dataset_root / "train" / "images"
    train_depth = dataset_root / "train" / "depth"
    val_rgb = dataset_root / "val" / "images"
    val_depth = dataset_root / "val" / "depth"
    test_rgb = dataset_root / "test" / "images"
    test_depth = dataset_root / "test" / "depth"

    train_ds = RGBDepthDataset(
        train_rgb, train_depth,
        image_size=image_size, is_train=True
    )
    
    val_ds = RGBDepthDataset(
        val_rgb, val_depth,
        image_size=image_size, is_train=False
    )
    
    test_ds = RGBDepthDataset(
        test_rgb, test_depth,
        image_size=image_size, is_train=False
    )
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Collate Function
# ============================================================================
def collate_esanet_depth_batch(batch):
    """
    ESANet RGB-Depth 배치 데이터를 안전하게 스택하는 함수입니다 (RGB 입력으로 Depth 예측).
    """
    rgbs = []
    depths = []
    filenames = []
    
    for sample in batch:
        if len(sample) >= 2:
            rgb, depth = sample[:2]
            rgbs.append(rgb.contiguous())
            depths.append(depth.contiguous())
            
            if len(sample) >= 3:
                filenames.append(sample[2])
        else:
            raise RuntimeError("Unexpected batch item length: expected >=2")
    
    rgbs = torch.stack(rgbs, dim=0)
    depths = torch.stack(depths, dim=0)
    
    if filenames:
        return rgbs, depths, filenames
    return rgbs, depths


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ESANet Depth Only")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml or config.json")
    parser.add_argument("--create-default-config", action="store_true", help="Create default config file and exit")
    return parser.parse_args()


def main() -> None:
    if not ESANET_AVAILABLE:
        print("❌ ESANet 모델을 사용할 수 없습니다.")
        return
    
    set_global_determinism(42)
    pl.seed_everything(42, workers=True)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    args = parse_args()
    
    if args.create_default_config:
        print("✅ Default configuration file creation not implemented for depth-only")
        return
    
    # 설정 파일 로드
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

    print(f"📋 Config type: {type(config)}")
    print(f"📋 Dataset root: {getattr(getattr(config, 'data', object()), 'dataset_root', 'N/A')}")
    
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    dataset_root = Path(os.path.abspath(config.data.dataset_root))
    output_dir = Path(os.path.abspath(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_start_time = time.time()

    train_ds, val_ds, test_ds = build_esanet_depth_datasets(
        dataset_root, (config.model.width, config.model.height)
    )
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_depth_batch,
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_depth_batch,
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_depth_batch,
    )

    steps_per_epoch = max(1, len(train_loader))
    
    vis_dir = config.visualization.vis_dir
    if vis_dir.strip().lower() == "none":
        vis_dir = ""
    elif vis_dir.strip() == "":
        vis_dir = str(output_dir / "vis")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

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

    model = LightningESANetDepthOnly(
        height=config.model.height,
        width=config.model.width,
        encoder=config.model.encoder,
        encoder_block=config.model.encoder_block,
        lr=config.training.lr,
        scheduler_t_max=getattr(config.training, 'scheduler_t_max', config.training.epochs * steps_per_epoch),
        final_lr=getattr(config.training, 'final_lr', 1e-5),
        loss_type=getattr(config.training, 'loss_type', 'silog'),
        save_vis_dir=vis_dir,
        vis_max=config.visualization.vis_max,
        save_root_dir=str(output_dir),
        pretrained_path=pretrained_path_to_use,
    )

    # AbsRel 기반 체크포인트
    ckpt_abs_rel = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="depth-abs-rel-{epoch:02d}-{val_abs_rel:.4f}",
        monitor="val_abs_rel",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    # 조기 종료
    es_monitor = getattr(config.training, 'early_stop_monitor', 'val_abs_rel')
    es_patience = config.training.early_stop_patience
    es_min_delta = config.training.early_stop_min_delta
    es_mode = 'min' if 'abs_rel' in str(es_monitor).lower() else 'max'
    
    early_stop = EarlyStopping(
        monitor=es_monitor,
        min_delta=es_min_delta,
        patience=es_patience,
        verbose=True,  # PyTorch Lightning이 직접 로그 출력
        mode=es_mode,
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="", version="")

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        precision=config.system.precision,
        default_root_dir=str(output_dir),
        callbacks=[ckpt_abs_rel, early_stop],
        logger=logger,
        accelerator=config.system.accelerator,
        devices=config.system.devices,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=(config.system.ckpt_path or None))
    
    print("=" * 80)
    print("Validating with best checkpoint (by lowest val_abs_rel)...")
    print("=" * 80)
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt_abs_rel.best_model_path)
    
    print("=" * 80)
    print("Testing with best checkpoint (by lowest val_abs_rel)...")
    print("=" * 80)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_abs_rel.best_model_path)

    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)
    
    summary = {
        "best_checkpoint_abs_rel": ckpt_abs_rel.best_model_path,
        "best_abs_rel": float(ckpt_abs_rel.best_model_score) if ckpt_abs_rel.best_model_score is not None else None,
        "best_val_results": best_val_results,
        "test_results": test_results,
        "training_time_sec": round(_elapsed_sec, 2),
        "training_time_hms": f"{_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d}",
    }

    print("=" * 80)
    print("학습완료!")
    print("=" * 80)
    print(summary)
    print(f"⏱️ 몇분이나 걸렸을까요?: {_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d} ({_elapsed_sec:.2f}s)")

    # CSV 저장
    try:
        csv_file = output_dir / "results.csv"
        fieldnames = [
            "training_time_sec", "training_time_hms",
            "best_abs_rel",
        ]
        
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            for k in best_val_results[0].keys():
                if k not in fieldnames:
                    fieldnames.append(f"val::{k}")
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            for k in test_results[0].keys():
                if k not in fieldnames:
                    fieldnames.append(f"test::{k}")

        row = {
            "training_time_sec": summary.get("training_time_sec", None),
            "training_time_hms": summary.get("training_time_hms", None),
            "best_abs_rel": summary.get("best_abs_rel", None),
        }
        
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            for k, v in best_val_results[0].items():
                row[f"val::{k}"] = v
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            for k, v in test_results[0].items():
                row[f"test::{k}"] = v

        with open(csv_file, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"🧾 CSV 결과가 저장되었습니다: {csv_file}")
    except Exception as e:
        warnings.warn(f"Failed to save CSV: {e}")

    # training.log 저장
    try:
        log_file = output_dir / "training.log"
        lines = []
        lines.append("=" * 80)
        lines.append("ESANet Depth-Only Learning Results")
        lines.append("=" * 80)
        lines.append(f"Training Time: {summary.get('training_time_hms', '')} ({summary.get('training_time_sec', 0)}s)")
        if summary.get("best_abs_rel") is not None:
            try:
                lines.append(f"Best AbsRel: {float(summary['best_abs_rel']):.4f}")
            except Exception:
                lines.append(f"Best AbsRel: {summary['best_abs_rel']}")
        if summary.get("best_checkpoint_abs_rel"):
            lines.append(f"Best AbsRel Checkpoint: {summary['best_checkpoint_abs_rel']}")
        lines.append("")
        
        if isinstance(best_val_results, list) and len(best_val_results) > 0 and isinstance(best_val_results[0], dict):
            lines.append("Validation Results:")
            for k, v in best_val_results[0].items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        
        if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
            lines.append("Test Results:")
            for k, v in test_results[0].items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        
        with open(log_file, "w", encoding="utf-8") as flog:
            flog.write("\n".join(lines))
        print(f"🗒️ training.log가 저장되었습니다: {log_file}")
    except Exception as e:
        warnings.warn(f"Failed to save training.log: {e}")


if __name__ == "__main__":
    main()
