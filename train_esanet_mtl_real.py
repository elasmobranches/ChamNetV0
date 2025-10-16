import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import time

# ESANet 모델 import
sys.path.append('/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet')
try:
    from src.models.model import ESANet
    ESANET_AVAILABLE = True
    print("✅ ESANet 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ ESANet 모델 import 실패: {e}")
    ESANET_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================
NUM_CLASSES = 7
ID2LABEL: Dict[int, str] = {
    0: "background",
    1: "chamoe",
    2: "heatpipe",
    3: "path",
    4: "pillar",
    5: "topdownfarm",
    6: "unknown",
}


# ============================================================================
# RGB+Depth Multi-Task Dataset (분리된 입력)
# ============================================================================
class RGBDepthMultiTaskDataset(Dataset):
    """
    RGB+Depth Multi-task dataset for Segmentation and Depth Estimation.
    ESANet은 RGB와 Depth를 분리된 입력으로 받습니다.
    Loads: RGB image, depth map, segmentation mask
    Returns: RGB tensor, depth tensor, segmentation mask, depth map
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
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

        # 이미지 파일 목록 가져오기
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
        
        # 파일명에서 확장자 제거하고 stem 추출
        img_stem = Path(img_name).stem
        
        # Mask path: {stem}_mask.png
        mask_name = f"{img_stem}_mask.png"
        mask_path = self.masks_dir / mask_name
        
        # Depth path: {stem}_depth.png  
        depth_name = f"{img_stem}_depth.png"
        depth_path = self.depth_dir / depth_name

        # 파일 존재 확인
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # Load data as PIL Images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        depth_img = Image.open(depth_path)
        
        # Convert depth to normalized float
        if depth_img.mode == 'I;16':
            depth = np.array(depth_img, dtype=np.float32) / 65535.0
        elif depth_img.mode == 'I':
            depth = np.array(depth_img, dtype=np.float32)
            if depth.max() > 1.0:
                depth = depth / depth.max()
        else:
            depth = np.array(depth_img.convert('L'), dtype=np.float32) / 255.0
        
        depth = Image.fromarray(depth)

        # Resize
        image = TF.resize(image, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.NEAREST)
        depth = TF.resize(depth, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)

        # Optional augmentation for training
        if self.is_train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                depth = TF.hflip(depth)

        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W] in [0, 1]
        depth_tensor = torch.from_numpy(np.array(depth, dtype=np.float32))  # [H, W]
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H, W]

        # Normalize image
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # ESANet은 RGB와 Depth를 분리된 텐서로 받음
        depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]

        return image, depth_tensor, mask, depth_tensor.squeeze(0), img_name  # RGB, Depth, Mask, Depth_target, filename


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    """Get preprocessing parameters for encoder"""
    # ESANet은 일반적으로 ImageNet normalization 사용
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================
# Loss Functions
# ============================================================================
class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss for depth estimation.
    """
    def __init__(self, lambda_variance: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.lambda_variance = lambda_variance
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, H, W] predicted depth
            target: [B, H, W] or [B, 1, H, W] ground truth depth
            mask: [B, H, W] or [B, 1, H, W] valid pixel mask (optional)
        """
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # Create mask for valid depths
        if mask is None:
            valid_mask = (target > self.eps) & (torch.isfinite(target))
        else:
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask = (target > self.eps) & (torch.isfinite(target)) & (mask > 0.5)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Apply mask
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute log diff
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        
        # SI-Log loss
        loss = (log_diff ** 2).mean() - self.lambda_variance * (log_diff.mean() ** 2)
        return loss


class L1DepthLoss(nn.Module):
    """L1 loss for depth estimation with valid mask handling"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        if mask is None:
            valid_mask = (target > self.eps) & (torch.isfinite(target))
        else:
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask = (target > self.eps) & (torch.isfinite(target)) & (mask > 0.5)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        return F.l1_loss(pred, target)


# ============================================================================
# ESANet-based Multi-Task Model
# ============================================================================
class ESANetMultiTask(nn.Module):
    """
    ESANet-based Multi-Task Learning for Segmentation and Depth Estimation.
    
    Architecture:
        - Shared Encoder: ESANet encoder (RGB+Depth 분리 입력)
        - Task-specific Heads: Segmentation head + Depth head
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
            pretrained_on_imagenet=False,  # ImageNet 사전 학습 비활성화 (나중에 NYUv2 가중치 로드)
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
        
        # 40개 클래스에서 7개 클래스로 변환하는 헤드 추가
        self.class_adapter = nn.Conv2d(40, num_classes, 1)
        
        # Depth estimation head 추가 (40개 클래스 출력에서)
        self.depth_head = nn.Sequential(
            nn.Conv2d(40, 64, 3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),  # BatchNorm 문제 해결
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),  # BatchNorm 문제 해결
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # 1채널 depth 출력
        )
        
        print(f"🔧 ESANet Multi-Task Architecture:")
        print(f"  - Input: RGB [B,3,H,W] + Depth [B,1,H,W] (separated)")
        print(f"  - Output: Segmentation ({num_classes} classes) + Depth (1 channel)")
        print(f"  - Encoder RGB: {encoder_rgb}")
        print(f"  - Encoder Depth: {encoder_depth}")
    
    def _fix_batchnorm_for_small_batches(self):
        """ESANet 내부의 BatchNorm을 작은 배치 크기에 맞게 수정"""
        def fix_batchnorm_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    # BatchNorm을 GroupNorm으로 교체
                    group_norm = nn.GroupNorm(num_groups=1, num_channels=child.num_features, 
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
        """안전한 사전 학습된 가중치 로드 - 호환되는 가중치만 로드"""
        try:
            # .pth 파일 직접 로드
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 체크포인트 키 매핑
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
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load pretrained weights: {e}")
            print("📝 Training from scratch...")
        
    def _load_pretrained_weights(self, pretrained_path: str):
        """사전 학습된 가중치 로드"""
        try:
            if pretrained_path.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(pretrained_path, 'r:gz') as tar:
                    # tar 파일에서 .pth 파일 추출
                    for member in tar.getmembers():
                        if member.name.endswith('.pth'):
                            # 임시 파일로 추출
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                                tar.extract(member, path=tempfile.gettempdir())
                                checkpoint_path = os.path.join(tempfile.gettempdir(), member.name)
                                break
            else:
                checkpoint_path = pretrained_path
            
            if checkpoint_path.endswith('.pth'):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 체크포인트 키 매핑 (필요시)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # ESANet 모델에 가중치 로드 (strict=False로 부분 매칭)
                self.esanet.load_state_dict(state_dict, strict=False)
                print("✅ Pretrained weights loaded successfully")
            else:
                print(f"⚠️ Warning: Unsupported checkpoint format: {checkpoint_path}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load pretrained weights: {e}")
            print("Training from scratch...")
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: [B, 3, H, W] RGB input tensor
            depth: [B, 1, H, W] Depth input tensor
        
        Returns:
            seg_logits: [B, num_classes, H, W] segmentation logits
            depth_pred: [B, 1, H, W] depth prediction (0-1 range)
        """
        # ESANet forward pass (40개 클래스 출력)
        esanet_output = self.esanet(rgb, depth)
        
        # ESANet이 훈련 모드에서 여러 출력을 반환하는 경우 처리
        if isinstance(esanet_output, tuple):
            esanet_features = esanet_output[0]  # 첫 번째 출력이 메인 segmentation 결과
        else:
            esanet_features = esanet_output
        
        # 40개 클래스에서 7개 클래스로 변환
        seg_logits = self.class_adapter(esanet_features)
        
        # Depth head forward pass (40개 클래스 특징에서)
        depth_raw = self.depth_head(esanet_features)
        
        # Apply sigmoid to constrain depth to [0, 1]
        depth_pred = torch.sigmoid(depth_raw)
        
        return seg_logits, depth_pred


# ============================================================================
# Depth Metrics
# ============================================================================
@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """
    Compute standard depth estimation metrics.
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    valid_mask = (target > eps) & (torch.isfinite(target)) & (torch.isfinite(pred))
    if valid_mask.sum() == 0:
        return {
            "abs_rel": torch.tensor(0.0, device=pred.device),
            "sq_rel": torch.tensor(0.0, device=pred.device),
            "rmse": torch.tensor(0.0, device=pred.device),
            "rmse_log": torch.tensor(0.0, device=pred.device),
            "delta1": torch.tensor(0.0, device=pred.device),
            "delta2": torch.tensor(0.0, device=pred.device),
            "delta3": torch.tensor(0.0, device=pred.device),
        }
    
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # AbsRel
    abs_rel = (torch.abs(pred - target) / (target + eps)).mean()

    # SqRel
    sq_rel = ((pred - target) ** 2 / (target + eps)).mean()

    # RMSE
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    
    # RMSE log
    rmse_log = torch.sqrt(((torch.log(pred + eps) - torch.log(target + eps)) ** 2).mean())
    
    # Threshold metrics
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
# PyTorch Lightning Module
# ============================================================================
class LightningESANetMTL(pl.LightningModule):
    """
    PyTorch Lightning module for ESANet Multi-Task Learning.
    
    Features:
        - RGB+Depth 분리 입력
        - ESANet backbone with pretrained weights
        - Multi-task learning (Segmentation + Depth)
        - Uncertainty-based loss weighting
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
        loss_type: str = "silog",  # "silog" or "l1"
        seg_loss_weight: float = 1.0,
        depth_loss_weight: float = 1.0,
        use_uncertainty_weighting: bool = True,
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = ESANetMultiTask(
            height=height,
            width=width,
            num_classes=num_classes,
            encoder_rgb=encoder_rgb,
            encoder_depth=encoder_depth,
            encoder_block=encoder_block,
            pretrained_path=pretrained_path,
        )
        
        # Loss functions
        # Segmentation: Cross-Entropy Loss
        self.seg_ce_loss = nn.CrossEntropyLoss()
        
        # Depth loss
        if loss_type == "silog":
            self.depth_loss_fn = SILogLoss()
        else:
            self.depth_loss_fn = L1DepthLoss()
        
        # Uncertainty-based weighting
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_depth = nn.Parameter(torch.zeros(1))
        else:
            self.seg_loss_weight = seg_loss_weight
            self.depth_loss_weight = depth_loss_weight
        
        self.num_classes = num_classes
        self.lr = lr
        self.t_max = scheduler_t_max
        self.base_vis_dir = save_vis_dir
        self.vis_max = int(vis_max) if vis_max is not None else 0
        self.save_root_dir = save_root_dir
        # track best miou for saving best visuals
        self._best_miou = float('-inf')
        # track best absrel (lower is better) for saving best depth visuals
        self._best_absrel = float('inf')
        
        # Curves storage
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
        self._epoch_train_metrics = []
        self._epoch_val_metrics = []
        
        # Class-wise IoU accumulation
        self._epoch_train_tp = torch.zeros(self.num_classes)
        self._epoch_train_fp = torch.zeros(self.num_classes)
        self._epoch_train_fn = torch.zeros(self.num_classes)
        self._epoch_val_tp = torch.zeros(self.num_classes)
        self._epoch_val_fp = torch.zeros(self.num_classes)
        self._epoch_val_fn = torch.zeros(self.num_classes)
        
        # Best values for adaptive normalization
        self.register_buffer("seg_best", torch.tensor(0.0))
        self.register_buffer("depth_best", torch.tensor(999.0))
        self.first_val_done = False
        
        # Visualization parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("vis_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("vis_std", torch.tensor(std).view(1, 3, 1, 1))
        
        # Color palette for segmentation
        self.register_buffer(
            "palette",
            torch.tensor([
                [0, 0, 0],       # 0 background
                [255, 255, 0],   # 1 chamoe
                [255, 0, 0],     # 2 heatpipe
                [0, 255, 0],     # 3 path
                [0, 0, 255],     # 4 pillar
                [255, 0, 255],   # 5 topdownfarm
                [128, 128, 128], # 6 unknown
            ], dtype=torch.uint8)
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(rgb, depth)
    
    def _compute_loss(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                     seg_target: torch.Tensor, depth_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-task loss with optional uncertainty weighting"""
        
        # Task losses
        seg_loss = self.seg_ce_loss(seg_logits, seg_target)
        depth_loss = self.depth_loss_fn(depth_pred, depth_target)
        
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting
            precision_seg = torch.exp(-self.log_var_seg)
            precision_depth = torch.exp(-self.log_var_depth)
            
            weighted_seg_loss = precision_seg * seg_loss + self.log_var_seg
            weighted_depth_loss = precision_depth * depth_loss + self.log_var_depth
            
            total_loss = weighted_seg_loss + weighted_depth_loss
        else:
            # Manual weighting
            weighted_seg_loss = self.seg_loss_weight * seg_loss
            weighted_depth_loss = self.depth_loss_weight * depth_loss
            total_loss = weighted_seg_loss + weighted_depth_loss
        
        loss_dict = {
            "total": total_loss,
            "seg": seg_loss,
            "depth": depth_loss,
            "weighted_seg": weighted_seg_loss,
            "weighted_depth": weighted_depth_loss,
        }
        
        if self.use_uncertainty_weighting:
            loss_dict["log_var_seg"] = self.log_var_seg
            loss_dict["log_var_depth"] = self.log_var_depth
        
        return total_loss, loss_dict

    @torch.no_grad()
    def _compute_metrics(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                        seg_target: torch.Tensor, depth_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute metrics for both tasks"""
        # Segmentation metrics
        prob = torch.softmax(seg_logits, dim=1)
        seg_pred = torch.argmax(prob, dim=1)
        
        # Simple accuracy calculation
        correct = (seg_pred == seg_target).float()
        seg_acc = correct.mean()
        
        # Class-wise IoU calculation
        tp = torch.zeros(self.num_classes, device=seg_pred.device)
        fp = torch.zeros(self.num_classes, device=seg_pred.device)
        fn = torch.zeros(self.num_classes, device=seg_pred.device)
        
        for c in range(self.num_classes):
            tp[c] = ((seg_pred == c) & (seg_target == c)).float().sum()
            fp[c] = ((seg_pred == c) & (seg_target != c)).float().sum()
            fn[c] = ((seg_pred != c) & (seg_target == c)).float().sum()
        
        # IoU calculation
        iou = tp / (tp + fp + fn + 1e-8)
        miou = iou.mean()
        
        # Depth metrics
        depth_metrics = compute_depth_metrics(depth_pred, depth_target)
        
        metrics = {
            "miou": miou,
            "seg_acc": seg_acc,
            "iou": iou,
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
        rgb, depth, seg_masks, depth_target = batch[:4]
        seg_logits, depth_pred = self(rgb, depth)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True)
        self.log("train_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True)
        self.log("train_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("train_abs_rel", metrics["abs_rel"], prog_bar=False, sync_dist=True)
        self.log("train_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        
        if self.use_uncertainty_weighting:
            self.log("log_var_seg", loss_dict["log_var_seg"], prog_bar=False, sync_dist=True)
            self.log("log_var_depth", loss_dict["log_var_depth"], prog_bar=False, sync_dist=True)
        
        # Store for epoch curves
        self._epoch_train_metrics.append({
            "loss": float(total_loss.detach().cpu().item()),
            "seg_loss": float(loss_dict["seg"].detach().cpu().item()),
            "depth_loss": float(loss_dict["depth"].detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
            "rmse_log": float(metrics["rmse_log"].detach().cpu().item()),
        })
        
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
        
        # FPS
        fps = float(rgb.shape[0]) / float(dt)
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True)
        self.log("val_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True)
        self.log("val_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("val_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True)
        self.log("val_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True)
        self.log("val_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        self.log("val_rmse_log", metrics["rmse_log"], prog_bar=False, sync_dist=True)
        self.log("val_delta1", metrics["delta1"], prog_bar=False, sync_dist=True)
        self.log("val_fps", fps, prog_bar=False, sync_dist=True)
        
        # No validation-stage visual saving
        
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
        
        # Store for epoch curves
        self._epoch_val_metrics.append({
            "loss": float(total_loss.detach().cpu().item()),
            "seg_loss": float(loss_dict["seg"].detach().cpu().item()),
            "depth_loss": float(loss_dict["depth"].detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
            "rmse_log": float(metrics["rmse_log"].detach().cpu().item()),
            "delta1": float(metrics["delta1"].detach().cpu().item()),
        })
        
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
        
        fps = float(rgb.shape[0]) / float(dt)
        
        # Logging
        self.log("test_loss", total_loss, sync_dist=True)
        self.log("test_seg_loss", loss_dict["seg"], sync_dist=True)
        self.log("test_depth_loss", loss_dict["depth"], sync_dist=True)
        self.log("test_miou", metrics["miou"], sync_dist=True)
        self.log("test_abs_rel", metrics["abs_rel"], sync_dist=True)
        self.log("test_sq_rel", metrics["sq_rel"], sync_dist=True)
        self.log("test_rmse", metrics["rmse"], sync_dist=True)
        self.log("test_rmse_log", metrics["rmse_log"], sync_dist=True)
        self.log("test_delta1", metrics["delta1"], sync_dist=True)
        self.log("test_acc", metrics["seg_acc"], sync_dist=True)
        self.log("test_fps", fps, sync_dist=True)
        
        # Save visuals for test: save for ALL batches (full dataset inference)
        self._maybe_save_visuals(rgb, depth, seg_masks, depth_target, seg_logits, depth_pred,
                                stage="test", batch_idx=batch_idx, filenames=filenames)
        
        return total_loss

    def on_train_epoch_end(self) -> None:
        if self._epoch_train_metrics:
            for key in ["loss", "seg_loss", "depth_loss", "miou", "abs_rel", "sq_rel", "rmse", "rmse_log"]:
                values = [m[key] for m in self._epoch_train_metrics if key in m]
                if values:
                    curve_key = f"train_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Reset accumulators
        self._epoch_train_metrics.clear()

    def on_validation_epoch_end(self) -> None:
        if self._epoch_val_metrics:
            for key in ["loss", "seg_loss", "depth_loss", "miou", "abs_rel", "sq_rel", "rmse", "rmse_log", "delta1"]:
                values = [m[key] for m in self._epoch_val_metrics if key in m]
                if values:
                    curve_key = f"val_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Reset accumulators
        self._epoch_val_metrics.clear()

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

            # Denormalize RGB images
            imgs = (rgb * self.vis_std + self.vis_mean).clamp(0, 1)
            imgs = (imgs * 255.0).to(torch.uint8).cpu()

            # Segmentation predictions
            seg_preds = torch.softmax(seg_logits, dim=1).argmax(dim=1).to(torch.int64).cpu()
            seg_gts = seg_masks.to(torch.int64).cpu()
            
            # Depth predictions and targets
            depth_preds = depth_pred.squeeze(1).cpu()
            depth_gts = depth_target.cpu()

            pal = self.palette.cpu()
            
            def colorize_seg(label_hw: torch.Tensor) -> np.ndarray:
                h, w = label_hw.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(min(self.num_classes, pal.shape[0])):
                    out[label_hw.numpy() == c] = pal[c].numpy()
                return out
            
            def colorize_depth(depth_hw: torch.Tensor) -> np.ndarray:
                """Colorize depth with viridis colormap"""
                depth_np = depth_hw.numpy()
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
                return (depth_colored * 255).astype(np.uint8)

            # For test stage, save all images in the batch; otherwise respect vis_max
            save_count = imgs.shape[0] if stage == "test" else min(self.vis_max, imgs.shape[0])
            for i in range(save_count):
                img = imgs[i].permute(1, 2, 0).numpy()
                
                # Segmentation results
                seg_gt = colorize_seg(seg_gts[i])
                seg_pr = colorize_seg(seg_preds[i])
                
                # Depth results
                depth_gt = colorize_depth(depth_gts[i])
                depth_pr = colorize_depth(depth_preds[i])
                
                # Resize depth to match image size if needed
                if depth_gt.shape[:2] != img.shape[:2]:
                    depth_gt = np.array(PILImage.fromarray(depth_gt).resize((img.shape[1], img.shape[0])))
                    depth_pr = np.array(PILImage.fromarray(depth_pr).resize((img.shape[1], img.shape[0])))
                
                # Create standardized 2x3 panel:
                # Row 1: [Seg_GT, Original, Seg_Pred]
                # Row 2: [Depth_GT, Original, Depth_Pred]
                row1 = np.concatenate([seg_gt, img, seg_pr], axis=1)
                row2 = np.concatenate([depth_gt, img, depth_pr], axis=1)
                panel = np.concatenate([row1, row2], axis=0)
                
                if filenames is not None and i < len(filenames):
                    stem = os.path.splitext(os.path.basename(filenames[i]))[0]
                    out_filename = f"{stem}.png"
                else:
                    out_filename = f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{i}.png"
                out_path = os.path.join(self.base_vis_dir, stage, out_filename)
                PILImage.fromarray(panel).save(out_path)
        except Exception as e:
            warnings.warn(f"Failed to save visualization: {e}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.t_max), eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ============================================================================
# Dataset Builder
# ============================================================================
def build_esanet_datasets(dataset_root: Path, image_size: Tuple[int, int]):
    """Build ESANet multi-task datasets"""
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "masks"
    train_depth = dataset_root / "train" / "depth"
    
    val_images = dataset_root / "val" / "images"
    val_masks = dataset_root / "val" / "masks"
    val_depth = dataset_root / "val" / "depth"
    
    test_images = dataset_root / "test" / "images"
    test_masks = dataset_root / "test" / "masks"
    test_depth = dataset_root / "test" / "depth"

    mean, std = get_preprocessing_params("esanet")

    train_ds = RGBDepthMultiTaskDataset(
        train_images, train_masks, train_depth,
        image_size=image_size, mean=mean, std=std, is_train=True
    )
    val_ds = RGBDepthMultiTaskDataset(
        val_images, val_masks, val_depth,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    test_ds = RGBDepthMultiTaskDataset(
        test_images, test_masks, test_depth,
        image_size=image_size, mean=mean, std=std, is_train=False
    )
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ESANet Multi-Task (Seg + Depth)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--precision", type=str, default="32", 
                       help="Training precision: 32, 16, bf16")
    parser.add_argument("--accelerator", type=str, default="auto", 
                       help="Accelerator: gpu/cpu/auto")
    parser.add_argument("--devices", type=str, default="auto", 
                       help="Devices: 1, 2, auto")
    parser.add_argument("--loss-type", type=str, default="silog", 
                       choices=["silog", "l1"], help="Depth loss type")
    parser.add_argument("--seg-loss-weight", type=float, default=1.0, 
                       help="Segmentation loss weight (if not using uncertainty weighting)")
    parser.add_argument("--depth-loss-weight", type=float, default=1.0, 
                       help="Depth loss weight (if not using uncertainty weighting)")
    parser.add_argument("--use-uncertainty-weighting", action="store_true", 
                       help="Use uncertainty-based loss weighting")
    parser.add_argument("--vis-dir", type=str, default="", 
                       help="Visualization directory")
    parser.add_argument("--vis-max", type=int, default=4, 
                       help="Max samples to visualize per epoch")
    parser.add_argument("--ckpt-path", type=str, default="", 
                       help="Resume from checkpoint")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, 
                       help="Gradient clipping value")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, 
                       help="Gradient accumulation steps")
    parser.add_argument("--pretrained-path", type=str, 
                       default="/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth", 
                       help="Path to pretrained ESANet weights")
    parser.add_argument("--encoder-rgb", type=str, default="resnet34", 
                       help="RGB encoder type")
    parser.add_argument("--encoder-depth", type=str, default="resnet34", 
                       help="Depth encoder type")
    parser.add_argument("--encoder-block", type=str, default="NonBottleneck1D", 
                       help="Encoder block type")
    
    return parser.parse_args()


def main() -> None:
    if not ESANET_AVAILABLE:
        print("❌ ESANet 모델을 사용할 수 없습니다.")
        return
    
    pl.seed_everything(42, workers=True)
    args = parse_args()
    
    # Print CUDA info
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    dataset_root = Path(os.path.abspath(args.dataset_root))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Measure total run time (train + val + test)
    _run_start_time = time.time()

    # Build datasets
    train_ds, val_ds, test_ds = build_esanet_datasets(
        dataset_root, (args.width, args.height)
    )
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    steps_per_epoch = max(1, len(train_loader))
    
    # Resolve visualization directory
    vis_dir = args.vis_dir
    if vis_dir.strip().lower() == "none":
        vis_dir = ""
    elif vis_dir.strip() == "":
        vis_dir = str(output_dir / "vis")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # Model
    model = LightningESANetMTL(
        height=args.height,
        width=args.width,
        num_classes=NUM_CLASSES,
        encoder_rgb=args.encoder_rgb,
        encoder_depth=args.encoder_depth,
        encoder_block=args.encoder_block,
        lr=args.lr,
        scheduler_t_max=args.epochs * steps_per_epoch,
        loss_type=args.loss_type,
        seg_loss_weight=args.seg_loss_weight,
        depth_loss_weight=args.depth_loss_weight,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        save_vis_dir=vis_dir,
        vis_max=args.vis_max,
        save_root_dir=str(output_dir),
        pretrained_path=args.pretrained_path if args.pretrained_path else None,
    )

    # Callbacks
    ckpt_miou = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="esanet-mtl-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,  # 최고 mIoU만 저장
        save_last=True,
    )
    
    ckpt_absrel = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="esanet-mtl-{epoch:02d}-{val_abs_rel:.4f}",
        monitor="val_abs_rel",
        mode="min",
        save_top_k=1,  # 최소 AbsRel만 저장
    )

    early_stop = EarlyStopping(
        monitor="val_abs_rel",
        min_delta=0.001,
        patience=30,
        verbose=True,
        mode="min",
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="esanet_mtl_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        default_root_dir=str(output_dir),
        callbacks=[ckpt_miou, ckpt_absrel, early_stop],
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
    )
    
    # Train
    # 훈련 시작
    trainer.fit(model, train_loader, val_loader, ckpt_path=(args.ckpt_path or None))
    
    # Validate with best checkpoint
    print("=" * 80)
    print("Validating with best checkpoint...")
    print("=" * 80)
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt_miou.best_model_path)
    
    # Test
    print("=" * 80)
    print("Testing...")
    print("=" * 80)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_miou.best_model_path)

    # Compute elapsed time
    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)
    
    # Summary
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
    
    print("=" * 80)
    print("학습완료!")
    print("=" * 80)
    print(summary)
    print(f"⏱️ 몇분이나 걸렸을까요?: {_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d} ({_elapsed_sec:.2f}s)")


if __name__ == "__main__":
    main()
