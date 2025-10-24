# ============================================================================
# ESANet 세그멘테이션 전용 학습 스크립트
# ============================================================================
# 이 스크립트는 ESANet 아키텍처를 기반으로 한 세그멘테이션 전용 학습을 수행합니다.
# 주요 기능:
# - RGB와 Depth를 분리된 입력으로 받는 ESANet 아키텍처
# - 세그멘테이션만 학습 (깊이 추정 헤드 제거)
# - 불확실성 기반 손실 가중치 조정 (단일 태스크용)
# - PyTorch Lightning을 활용한 효율적인 학습 관리

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
from tqdm import tqdm

# torchmetrics 라이브러리 import (효율적인 메트릭 계산을 위해)
try:
    from torchmetrics import JaccardIndex, MeanSquaredError, MeanAbsoluteError
    TORCHMETRICS_AVAILABLE = True
    print("✅ torchmetrics를 성공적으로 import했습니다.")
except ImportError:
    print("⚠️ torchmetrics가 설치되지 않았습니다. pip install torchmetrics를 실행하세요.")
    TORCHMETRICS_AVAILABLE = False

# PyTorch Lightning 관련 import
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
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
NUM_CLASSES = 7

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
# RGB+Depth 세그멘테이션 전용 데이터셋
# ============================================================================
class RGBSegmentationDataset(Dataset):
    """
    세그멘테이션 전용 RGB 데이터셋입니다.
    """
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size: Tuple[int, int],
        mean: List[float],
        std: List[float],
        is_train: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

        self.image_files: List[str] = [
            f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images in {images_dir}")

        print(f"📁 Dataset loaded: {len(self.image_files)} images from {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        img_stem = Path(img_name).stem
        mask_name = f"{img_stem}_mask.png"
        mask_path = self.masks_dir / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # PIL 이미지로 데이터 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 모든 이미지를 모델 입력 크기로 리사이즈
        image = TF.resize(image, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.NEAREST)

        # PIL 이미지를 PyTorch 텐서로 변환
        image = TF.to_tensor(image).contiguous().clone()
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).contiguous().clone()

        # ImageNet 표준으로 RGB 이미지 정규화
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, mask, img_name


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================
# ESANet 세그멘테이션 전용 모델
# ============================================================================
class ESANetRGBOnly(nn.Module):
    """
    ESANet One Modality 구조를 사용한 RGB 전용 모델입니다.
    """
    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        num_classes: int = 7,
        encoder_rgb: str = 'resnet34',
        encoder_block: str = 'NonBottleneck1D',
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        if not ESANET_AVAILABLE:
            raise ImportError("ESANet One Modality 모델을 사용할 수 없습니다.")
        
        # ESANet One Modality 모델 초기화 (RGB 전용)
        self.esanet = ESANetOneModality(
            height=height,
            width=width,
            num_classes=num_classes,  # 직접 7개 클래스로 초기화
            encoder=encoder_rgb,
            encoder_block=encoder_block,
            pretrained_on_imagenet=False,
            activation='relu',
            input_channels=3,  # RGB 입력
            encoder_decoder_fusion='add',
            context_module='ppm',
            upsampling='bilinear',
        )
        
        # ESANet 내부의 BatchNorm 설정 변경
        self._fix_batchnorm_for_small_batches()
        
        # 사전 학습된 가중치 로드 (40클래스 → 7클래스 변환 필요)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading pretrained ESANet weights from {pretrained_path}")
            self._load_pretrained_weights_safe(pretrained_path)
        else:
            print("📝 No pretrained weights provided, training from scratch...")
        
        # 어댑터 제거 (직접 7개 클래스 출력)
        self.class_adapter = None
        
        print(f"🔧 ESANet RGB-Only Architecture (One Modality):")
        print(f"  - Input: RGB [B,3,H,W] only")
        print(f"  - Output: Segmentation ({num_classes} classes)")
        print(f"  - Encoder: {encoder_rgb} (RGB only)")
        print(f"  - Depth encoder/decoder: COMPLETELY REMOVED")
    
    def _fix_batchnorm_for_small_batches(self):
        def fix_batchnorm_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_groups = min(32, child.num_features)
                    if child.num_features % num_groups != 0:
                        num_groups = 1
                    
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
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load pretrained weights: {e}")
            print("📝 Training from scratch...")
        
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        ESANet RGB 전용 모델의 순전파 연산 (완전한 RGB 전용)
        
        Args:
            rgb: RGB 입력 텐서 [B, 3, H, W]
        
        Returns:
            seg_logits: 세그멘테이션 로짓 [B, num_classes, H, W]
        """
        # ESANet One Modality 순전파 (RGB만 사용)
        esanet_output = self.esanet(rgb)
        
        # ESANet이 훈련 모드에서 여러 출력을 반환하는 경우 처리
        if isinstance(esanet_output, tuple):
            seg_logits = esanet_output[0]
        else:
            seg_logits = esanet_output
        
        # 직접 7개 클래스 출력 (어댑터 불필요)
        return seg_logits


# ============================================================================
# Custom Early Stopping Callback with Accurate Logging
# ============================================================================
class CustomEarlyStopping(EarlyStopping):
    """
    정확한 Early Stopping 로그를 출력하는 커스텀 콜백입니다.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_logged_wait_count = -1
    
    def on_validation_end(self, trainer, pl_module):
        """validation 끝에서 정확한 로그 출력"""
        # EarlyStopping 로직 먼저 실행 (wait_count 업데이트)
        super().on_validation_end(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """validation epoch 끝에서 정확한 로그 출력"""
        # 모든 validation epoch이 끝난 후 로그 출력
        current_wait = getattr(self, 'wait_count', 0)
        patience = getattr(self, 'patience', 0)
        monitor = getattr(self, 'monitor', 'val_loss')
        
        # 로그 출력 (tqdm을 사용하여 진행률 표시줄과 함께 출력)
        from tqdm import tqdm
        tqdm.write(
            f"Epoch {trainer.current_epoch:3d}: "
            f"Val Loss: {trainer.callback_metrics.get('val_loss', 0.0):.4f}, "
            f"Val mIoU: {trainer.callback_metrics.get('val_miou', 0.0):.4f} | "
            f"Early stop({monitor}): {current_wait}/{patience}"
        )
        
        self._last_logged_wait_count = current_wait


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
class LightningESANetSegOnly(pl.LightningModule):
    """
    ESANet 세그멘테이션 전용 학습을 위한 PyTorch Lightning 모듈입니다.
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
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # ESANet RGB 전용 모델 초기화
        self.model = ESANetRGBOnly(
            height=height,
            width=width,
            num_classes=num_classes,
            encoder_rgb=encoder_rgb,
            encoder_block=encoder_block,
            pretrained_path=pretrained_path,
        )
        
        # 손실 함수 설정
        try:
            import segmentation_models_pytorch as smp
            self.seg_dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            self.seg_ce_loss = nn.CrossEntropyLoss()
            self._use_dice = True
        except Exception:
            self.seg_ce_loss = nn.CrossEntropyLoss()
            self._use_dice = False
        
        # 기본 설정 저장
        self.num_classes = num_classes
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
            "train_miou": [],
            "val_miou": [],
        }
        
        # 메모리 효율적인 메트릭 누적기
        self._train_metrics_accumulator = None
        self._val_metrics_accumulator = None
        
        # 수동 계산을 위한 에폭 메트릭 초기화
        self._epoch_train_tp = None
        self._epoch_train_fp = None
        self._epoch_train_fn = None
        self._epoch_val_tp = None
        self._epoch_val_fp = None
        self._epoch_val_fn = None
        
        # 효율적인 메트릭 계산을 위한 torchmetrics 설정
        if TORCHMETRICS_AVAILABLE:
            self.train_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
            self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')
        else:
            self._epoch_train_tp = None
            self._epoch_train_fp = None
            self._epoch_train_fn = None
            self._epoch_val_tp = None
            self._epoch_val_fp = None
            self._epoch_val_fn = None
        
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
        # logged_metrics removed - using PyTorch Lightning's callback_metrics instead

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.model(rgb)
    
    def _compute_loss(self, seg_logits: torch.Tensor, seg_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        세그멘테이션 손실 계산
        """
        if self._use_dice:
            seg_dice = self.seg_dice_loss(seg_logits, seg_target)
            seg_ce = self.seg_ce_loss(seg_logits, seg_target)
            seg_loss = 0.7 * seg_dice + 0.3 * seg_ce
        else:
            seg_dice = None
            seg_ce = self.seg_ce_loss(seg_logits, seg_target)
            seg_loss = seg_ce
        
        loss_dict = {
            "total": seg_loss,
            "seg": seg_loss,
            "seg_dice": seg_dice if seg_dice is not None else torch.tensor(0.0, device=seg_logits.device),
            "seg_ce": seg_ce,
        }
        
        return seg_loss, loss_dict

    @torch.no_grad()
    def _compute_metrics(self, seg_logits: torch.Tensor, seg_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        세그멘테이션 메트릭 계산
        """
        prob = torch.softmax(seg_logits, dim=1)
        seg_pred = torch.argmax(prob, dim=1)
        
        correct = (seg_pred == seg_target).float()
        seg_acc = correct.mean()
        
        if TORCHMETRICS_AVAILABLE:
            if seg_pred.device != next(self.parameters()).device:
                seg_pred = seg_pred.to(next(self.parameters()).device)
                seg_target = seg_target.to(next(self.parameters()).device)
            
            miou = self.train_iou(seg_pred, seg_target) if self.training else self.val_iou(seg_pred, seg_target)
            
            metrics = {
                "miou": miou,
                "seg_acc": seg_acc,
            }
        else:
            # 수동 계산으로 폴백
            tp = torch.zeros(self.num_classes, device=seg_pred.device)
            fp = torch.zeros(self.num_classes, device=seg_pred.device)
            fn = torch.zeros(self.num_classes, device=seg_pred.device)
            
            for c in range(self.num_classes):
                tp[c] = ((seg_pred == c) & (seg_target == c)).float().sum()
                fp[c] = ((seg_pred == c) & (seg_target != c)).float().sum()
                fn[c] = ((seg_pred != c) & (seg_target == c)).float().sum()
            
            iou = tp / (tp + fp + fn + 1e-8)
            miou = iou.mean()
            
            metrics = {
                "miou": miou,
                "seg_acc": seg_acc,
            }
        
        return metrics

    def training_step(self, batch, batch_idx):
        rgb, seg_masks = batch[:2]
        seg_logits = self(rgb)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, seg_masks)
        metrics = self._compute_metrics(seg_logits.detach(), seg_masks)
        
        # 메트릭 로깅
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("train_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        if self._use_dice:
            self.log("train_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("train_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("train_miou", metrics["miou"], prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, batch_size=rgb.shape[0])
        
        # 에폭 레벨 IoU 계산을 위한 클래스별 통계 누적
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        
        if self._epoch_train_tp is None:
            device = seg_pred.device
            self._epoch_train_tp = torch.zeros(self.num_classes, device=device)
            self._epoch_train_fp = torch.zeros(self.num_classes, device=device)
            self._epoch_train_fn = torch.zeros(self.num_classes, device=device)
        
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            self._epoch_train_tp[c] += tp
            self._epoch_train_fp[c] += fp
            self._epoch_train_fn[c] += fn

        if self._train_metrics_accumulator is None:
            self._train_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        metrics_to_store = {
            "loss": total_loss,
            "seg_loss": loss_dict["seg"],
            "seg_ce": loss_dict["seg_ce"],
            "miou": metrics["miou"],
        }
        
        if self._use_dice and "seg_dice" in loss_dict:
            metrics_to_store["seg_dice"] = loss_dict["seg_dice"]
        
        self._train_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb, seg_masks = batch[:2]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg_logits = self(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, seg_masks)
        metrics = self._compute_metrics(seg_logits.detach(), seg_masks)
        
        # FPS
        fps = float(rgb.shape[0]) / float(dt)
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        if self._use_dice:
            self.log("val_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("val_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("val_miou", metrics["miou"], prog_bar=True, sync_dist=True, batch_size=rgb.shape[0])
        self.log("val_fps", fps, prog_bar=False, sync_dist=True, batch_size=rgb.shape[0])
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        
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

        # Store for epoch summary logging (removed - using PyTorch Lightning's callback_metrics instead)
        
        if self._val_metrics_accumulator is None:
            self._val_metrics_accumulator = MetricsAccumulator(device=total_loss.device)
        
        metrics_to_store = {
            "loss": total_loss,
            "seg_loss": loss_dict["seg"],
            "seg_ce": loss_dict["seg_ce"],
            "miou": metrics["miou"],
        }
        
        if self._use_dice and "seg_dice" in loss_dict:
            metrics_to_store["seg_dice"] = loss_dict["seg_dice"]
        
        self._val_metrics_accumulator.update(**metrics_to_store)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        rgb, seg_masks = batch[:2]
        filenames = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            filenames = batch[2]
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg_logits = self(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, seg_masks)
        metrics = self._compute_metrics(seg_logits.detach(), seg_masks)
        
        fps = float(rgb.shape[0]) / float(dt)
        
        # Logging
        self.log("test_loss", total_loss, sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_seg_loss", loss_dict["seg"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_miou", metrics["miou"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_acc", metrics["seg_acc"], sync_dist=True, batch_size=rgb.shape[0])
        self.log("test_fps", fps, sync_dist=True, batch_size=rgb.shape[0])
        
        # 클래스별 IoU 집계를 위한 통계 누적
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
        
        # Save visuals for test
        self._maybe_save_visuals(rgb, seg_masks, seg_logits,
                                stage="test", batch_idx=batch_idx, filenames=filenames)
        
        return total_loss

    def on_test_epoch_end(self) -> None:
        """테스트 에폭 종료 시 클래스별 IoU를 로그로 기록"""
        try:
            if hasattr(self, "_epoch_test_tp") and self._epoch_test_tp is not None:
                class_names = [
                    "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
                ]
                for i, class_name in enumerate(class_names[: self.num_classes]):
                    denom = (self._epoch_test_tp[i] + self._epoch_test_fp[i] + self._epoch_test_fn[i])
                    iou_value = (self._epoch_test_tp[i] / denom) if denom > 0 else torch.tensor(0.0, device=self._epoch_test_tp.device)
                    self.log(f"test_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)
        finally:
            if hasattr(self, "_epoch_test_tp") and self._epoch_test_tp is not None:
                self._epoch_test_tp = None
                self._epoch_test_fp = None
                self._epoch_test_fn = None

    def on_train_epoch_end(self) -> None:
        if self._train_metrics_accumulator is not None:
            epoch_metrics = self._train_metrics_accumulator.compute_mean()
            
            for key, value in epoch_metrics.items():
                curve_key = f"train_{key}"
                if curve_key in self.curves:
                    self.curves[curve_key].append(value)
            
            self._train_metrics_accumulator.reset()
        
        if TORCHMETRICS_AVAILABLE:
            epoch_iou = self.train_iou.compute()
            self.log("train_epoch_iou", epoch_iou, prog_bar=True, sync_dist=True)
            self.train_iou.reset()
        else:
            if self._epoch_train_tp is not None:
                class_names = [
                    "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
                ]
                for i, class_name in enumerate(class_names[: self.num_classes]):
                    denom = (self._epoch_train_tp[i] + self._epoch_train_fp[i] + self._epoch_train_fn[i])
                    iou_value = (self._epoch_train_tp[i] / denom) if denom > 0 else torch.tensor(0.0)
                    self.log(f"train_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)

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
            # Compute epoch-level IoU
            epoch_iou = self.val_iou.compute()
            self.log("val_epoch_iou", epoch_iou, prog_bar=True, sync_dist=True)
            # cache for summary print
            try:
                self._last_val_epoch_iou = float(epoch_iou.detach().cpu().item())
            except Exception:
                self._last_val_epoch_iou = None
            self.val_iou.reset()
        else:
            # Fallback to manual computation
            if self._epoch_val_tp is not None:
                class_names = [
                    "background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"
                ]
                for i, class_name in enumerate(class_names[: self.num_classes]):
                    denom = (self._epoch_val_tp[i] + self._epoch_val_fp[i] + self._epoch_val_fn[i])
                    iou_value = (self._epoch_val_tp[i] / denom) if denom > 0 else torch.tensor(0.0)
                    self.log(f"val_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)

        
        # 로그 출력은 EarlyStopping 콜백(verbose=True)에서 처리됩니다

        # Reset accumulators
        if self._epoch_val_tp is not None:
            self._epoch_val_tp.zero_()
            self._epoch_val_fp.zero_()
            self._epoch_val_fn.zero_()

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
            
            # mIoU
            ax = axes[1]
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
            
        except Exception as e:
            warnings.warn(f"Failed to save curves: {e}")

    @torch.no_grad()
    def _maybe_save_visuals(self, rgb: torch.Tensor, seg_masks: torch.Tensor, 
                           seg_logits: torch.Tensor, stage: str, batch_idx: int,
                           filenames: Optional[List[str]] = None) -> None:
        if not self.base_vis_dir or self.vis_max <= 0:
            return
        if stage != "test" and batch_idx != 0:
            return
        
        try:
            import os
            from PIL import Image as PILImage
            os.makedirs(os.path.join(self.base_vis_dir, stage), exist_ok=True)

            # Denormalize RGB images
            imgs = (rgb * self.vis_std + self.vis_mean).clamp(0, 1)
            imgs = (imgs * 255.0).to(torch.uint8)

            # Segmentation predictions
            seg_preds = torch.softmax(seg_logits, dim=1).argmax(dim=1).to(torch.int64)
            seg_gts = seg_masks.to(torch.int64)

            pal = self.palette.cpu()
            
            def colorize_seg(label_hw: torch.Tensor) -> np.ndarray:
                label_np = label_hw.detach().cpu().numpy()
                h, w = label_np.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(min(self.num_classes, pal.shape[0])):
                    out[label_np == c] = pal[c].numpy()
                return out

            save_count = imgs.shape[0] if stage == "test" else min(self.vis_max, imgs.shape[0])
            
            for i in range(save_count):
                img = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
                
                seg_gt = colorize_seg(seg_gts[i])
                seg_pr = colorize_seg(seg_preds[i])
                
                # Create 1x3 panel
                panel = np.concatenate([seg_gt, img, seg_pr], axis=1)
                
                if filenames is not None and i < len(filenames):
                    stem = os.path.splitext(os.path.basename(filenames[i]))[0]
                    out_filename = f"{stem}.png"
                else:
                    out_filename = f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{i}.png"
                out_path = os.path.join(self.base_vis_dir, stage, out_filename)
                Image.fromarray(panel).save(out_path)
                
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
def build_esanet_seg_datasets(dataset_root: Path, image_size: Tuple[int, int]):
    """
    ESANet 세그멘테이션 전용 학습을 위한 데이터셋을 구축합니다 (RGB 전용).
    """
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "masks"
    
    val_images = dataset_root / "val" / "images"
    val_masks = dataset_root / "val" / "masks"
    
    test_images = dataset_root / "test" / "images"
    test_masks = dataset_root / "test" / "masks"

    mean, std = get_preprocessing_params("esanet")

    train_ds = RGBSegmentationDataset(
        train_images, train_masks,
        image_size=image_size, mean=mean, std=std, is_train=True
    )
    
    val_ds = RGBSegmentationDataset(
        val_images, val_masks,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    
    test_ds = RGBSegmentationDataset(
        test_images, test_masks,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Collate Function
# ============================================================================
def collate_esanet_seg_batch(batch):
    """
    ESANet 세그멘테이션 전용 배치 데이터를 안전하게 스택하는 함수입니다 (RGB 전용).
    """
    imgs = []
    masks = []
    filenames = []
    
    for sample in batch:
        if len(sample) >= 2:
            img, mask = sample[:2]
            imgs.append(img.contiguous())
            masks.append(mask.contiguous())
            
            if len(sample) >= 3:
                filenames.append(sample[2])
        else:
            raise RuntimeError("Unexpected batch item length: expected >=2")
    
    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)
    
    if filenames:
        return imgs, masks, filenames
    return imgs, masks


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ESANet Segmentation Only")
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
        print("✅ Default configuration file creation not implemented for seg-only")
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

    train_ds, val_ds, test_ds = build_esanet_seg_datasets(
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
        collate_fn=collate_esanet_seg_batch,
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_seg_batch,
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        worker_init_fn=dataloader_worker_init_fn,
        collate_fn=collate_esanet_seg_batch,
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

    model = LightningESANetSegOnly(
        height=config.model.height,
        width=config.model.width,
        num_classes=config.model.num_classes,
        encoder_rgb=config.model.encoder_rgb,
        encoder_block=config.model.encoder_block,
        lr=config.training.lr,
        scheduler_t_max=getattr(config.training, 'scheduler_t_max', config.training.epochs * steps_per_epoch),
        final_lr=getattr(config.training, 'final_lr', 1e-5),
        save_vis_dir=vis_dir,
        vis_max=config.visualization.vis_max,
        save_root_dir=str(output_dir),
        pretrained_path=pretrained_path_to_use,
    )

    # mIoU 기반 체크포인트
    ckpt_miou = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="rgb-seg-miou-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    # 조기 종료
    es_monitor = getattr(config.training, 'early_stop_monitor', 'val_miou')
    es_patience = config.training.early_stop_patience
    es_min_delta = config.training.early_stop_min_delta
    es_mode = 'max' if 'iou' in str(es_monitor).lower() else 'min'
    
    # 방법 1: CustomEarlyStopping 사용 (정확한 로그)
    # early_stop = CustomEarlyStopping(
    #     monitor=es_monitor,
    #     min_delta=es_min_delta,
    #     patience=es_patience,
    #     verbose=False,  # CustomEarlyStopping에서 로그 출력하므로 False
    #     mode=es_mode,
    # )
    
    # 방법 2: 간단한 해결책 (현재 사용)
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
        callbacks=[ckpt_miou, early_stop],
        logger=logger,
        accelerator=config.system.accelerator,
        devices=config.system.devices,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=(config.system.ckpt_path or None))
    
    print("=" * 80)
    print("Validating with best checkpoint (by highest val_miou)...")
    print("=" * 80)
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt_miou.best_model_path)
    
    print("=" * 80)
    print("Testing with best checkpoint (by highest val_miou)...")
    print("=" * 80)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_miou.best_model_path)

    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)
    
    summary = {
        "best_checkpoint_miou": ckpt_miou.best_model_path,
        "best_miou": float(ckpt_miou.best_model_score) if ckpt_miou.best_model_score is not None else None,
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
            "best_miou",
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
            "best_miou": summary.get("best_miou", None),
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
        lines.append("ESANet RGB-Only Segmentation Learning Results")
        lines.append("=" * 80)
        lines.append(f"Training Time: {summary.get('training_time_hms', '')} ({summary.get('training_time_sec', 0)}s)")
        if summary.get("best_miou") is not None:
            try:
                lines.append(f"Best mIoU: {float(summary['best_miou']):.4f}")
            except Exception:
                lines.append(f"Best mIoU: {summary['best_miou']}")
        if summary.get("best_checkpoint_miou"):
            lines.append(f"Best mIoU Checkpoint: {summary['best_checkpoint_miou']}")
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
