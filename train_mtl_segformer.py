import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
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
# RGB+Depth Multi-Task Dataset (Early Fusion)
# ============================================================================
class RGBDepthMultiTaskDataset(Dataset):
    """
    RGB+Depth Multi-task dataset for Segmentation and Depth Estimation.
    SegFormer는 RGB와 Depth를 4채널 입력으로 Early Fusion합니다.
    Loads: RGB image, depth map, segmentation mask
    Returns: RGBD tensor (4 channels), segmentation mask, depth map
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
        
        # SegFormer Early Fusion: RGB + Depth를 4채널로 결합
        depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]
        rgbd_tensor = torch.cat([image, depth_tensor], dim=0)  # [4, H, W] - RGBD

        return rgbd_tensor, mask, depth_tensor.squeeze(0)  # RGBD, Mask, Depth_target


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    """Get preprocessing parameters for encoder"""
    # SegFormer는 ImageNet normalization 사용
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
# SegFormer-based Multi-Task Model (Early Fusion)
# ============================================================================
class HardSharingSegformerMTL(nn.Module):
    """
    Hard Parameter Sharing Multi-Task Learning with SegFormer.
    
    Architecture:
        - Shared Encoder: SegFormer encoder (RGB+Depth Early Fusion - 4채널 입력)
        - Task-specific Decoder 1: Segmentation head
        - Task-specific Decoder 2: Depth estimation head
    """
    def __init__(
        self,
        encoder_name: str = "mit_b2",
        num_classes: int = 7,
        depth_channels: int = 1,
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()
        # 4ch RGBD 입력을 3ch로 투영하는 1x1 컨볼루션 (사전학습 인코더 사용을 위함)
        self.rgbd_to_rgb = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        # 초기화: RGB 채널은 identity, Depth 채널 가중치는 0으로 시작
        with torch.no_grad():
            self.rgbd_to_rgb.weight.zero_()
            # weight shape: [3, 4, 1, 1]
            self.rgbd_to_rgb.weight[0, 0, 0, 0] = 1.0  # R
            self.rgbd_to_rgb.weight[1, 1, 0, 0] = 1.0  # G
            self.rgbd_to_rgb.weight[2, 2, 0, 0] = 1.0  # B
            # Depth 채널(입력 index=3)은 작게 기여하도록 초기값 부여
            self.rgbd_to_rgb.weight[:, 3, 0, 0] = 0.1

        # Create segmentation model (provides encoder + decoder)
        self.seg_model = smp.create_model(
            arch="Segformer",
            encoder_name=encoder_name,
            in_channels=3,  # 사전학습 인코더 유지
            classes=num_classes,
            encoder_weights=encoder_weights,
        )
        
        # Create depth model (shared encoder, separate decoder)
        self.depth_model = smp.create_model(
            arch="Segformer",
            encoder_name=encoder_name,
            in_channels=3,  # 사전학습 인코더 유지
            classes=depth_channels,
            encoder_weights=encoder_weights,
        )
        
        # Share encoder parameters between tasks
        # Replace depth encoder with segmentation encoder (hard parameter sharing)
        self.shared_encoder = self.seg_model.encoder
        self.depth_model.encoder = self.shared_encoder
        
        # Task-specific decoders
        self.seg_decoder = self.seg_model.decoder
        self.seg_head = self.seg_model.segmentation_head
        
        self.depth_decoder = self.depth_model.decoder
        self.depth_head = self.depth_model.segmentation_head
        
        print(f"🔧 SegFormer Multi-Task Architecture (RGB+Depth):")
        print(f"  - Input: RGBD [B,4,H,W] → Conv1x1 → [B,3,H,W] → Encoder")
        print(f"  - Output: Segmentation ({num_classes} classes) + Depth (1 channel)")
        print(f"  - Encoder: {encoder_name}")
        print(f"  - Shared Encoder: ✅")
    
    def forward(self, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgbd: [B, 4, H, W] RGBD input tensor (RGB + Depth)
        
        Returns:
            seg_logits: [B, num_classes, H, W] segmentation logits
            depth_pred: [B, 1, H, W] depth prediction (0-1 range)
        """
        # Project 4ch RGBD to 3ch using 1x1 conv, then shared encoder
        projected_rgb = self.rgbd_to_rgb(rgbd)
        features = self.shared_encoder(projected_rgb)
        
        # Task-specific decoders
        # Note: decoder expects features as a list/tuple, not unpacked
        seg_features = self.seg_decoder(features)
        seg_logits = self.seg_head(seg_features)
        
        depth_features = self.depth_decoder(features)
        depth_raw = self.depth_head(depth_features)
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
class LightningSegformerMTL(pl.LightningModule):
    """
    PyTorch Lightning module for SegFormer Multi-Task Learning.
    
    Features:
        - RGB+Depth Early Fusion (4채널 입력)
        - SegFormer backbone with pretrained weights
        - Multi-task learning (Segmentation + Depth)
        - Uncertainty-based loss weighting
    """
    def __init__(
        self,
        encoder_name: str = "mit_b2",
        num_classes: int = 7,
        lr: float = 1e-4,
        scheduler_t_max: int = 1000,
        loss_type: str = "silog",  # "silog" or "l1"
        seg_loss_weight: float = 1.0,
        depth_loss_weight: float = 1.0,
        use_uncertainty_weighting: bool = True,
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
        encoder_weights: Optional[str] = "imagenet",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = HardSharingSegformerMTL(
            encoder_name=encoder_name,
            num_classes=num_classes,
            encoder_weights=encoder_weights,
        )
        
        # Loss functions
        # Segmentation: Combined Dice + Cross-Entropy for better class balance
        self.seg_dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
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
        
        # Curves storage
        self.curves = {
            "train_loss": [],
            "val_loss": [],
            "train_seg_loss": [],
            "val_seg_loss": [],
            "train_seg_dice": [],
            "val_seg_dice": [],
            "train_seg_ce": [],
            "val_seg_ce": [],
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
            "val_delta1": [],
        }
        self._epoch_train_metrics = []
        self._epoch_val_metrics = []
        
        # Class-wise IoU accumulation
        self.register_buffer("_epoch_train_tp", torch.zeros(self.num_classes))
        self.register_buffer("_epoch_train_fp", torch.zeros(self.num_classes))
        self.register_buffer("_epoch_train_fn", torch.zeros(self.num_classes))
        self.register_buffer("_epoch_val_tp", torch.zeros(self.num_classes))
        self.register_buffer("_epoch_val_fp", torch.zeros(self.num_classes))
        self.register_buffer("_epoch_val_fn", torch.zeros(self.num_classes))
        
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

    def forward(self, rgbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(rgbd)
    
    def _compute_loss(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                     seg_target: torch.Tensor, depth_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-task loss with optional uncertainty weighting"""
        
        # Task losses
        # Segmentation: Combined Dice + Cross-Entropy
        seg_dice = self.seg_dice_loss(seg_logits, seg_target)
        seg_ce = self.seg_ce_loss(seg_logits, seg_target)
        seg_loss = 0.7 * seg_dice + 0.3 * seg_ce  # Weighted combination
        
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
            "seg_dice": seg_dice,
            "seg_ce": seg_ce,
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
        rgbd, seg_masks, depth_target = batch
        seg_logits, depth_pred = self(rgbd)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True)
        self.log("train_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("train_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("train_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True)
        self.log("train_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("train_abs_rel", metrics["abs_rel"], prog_bar=False, sync_dist=True)
        self.log("train_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True)
        self.log("train_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        self.log("train_rmse_log", metrics["rmse_log"], prog_bar=False, sync_dist=True)
        
        if self.use_uncertainty_weighting:
            self.log("log_var_seg", loss_dict["log_var_seg"], prog_bar=False, sync_dist=True)
            self.log("log_var_depth", loss_dict["log_var_depth"], prog_bar=False, sync_dist=True)
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            
            self._epoch_train_tp[c] += tp
            self._epoch_train_fp[c] += fp
            self._epoch_train_fn[c] += fn
        
        # Store for epoch curves
        self._epoch_train_metrics.append({
            "loss": float(total_loss.detach().cpu().item()),
            "seg_loss": float(loss_dict["seg"].detach().cpu().item()),
            "seg_dice": float(loss_dict["seg_dice"].detach().cpu().item()),
            "seg_ce": float(loss_dict["seg_ce"].detach().cpu().item()),
            "depth_loss": float(loss_dict["depth"].detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
        })
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgbd, seg_masks, depth_target = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                seg_logits, depth_pred = self(rgbd)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            seg_logits, depth_pred = self(rgbd)
            dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # FPS and per-image latency (ms)
        batch_size = float(rgbd.shape[0])
        fps = batch_size / float(dt)
        val_latency_ms = (float(dt) / max(1.0, batch_size)) * 1000.0
        
        # Logging
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_seg_loss", loss_dict["seg"], prog_bar=True, sync_dist=True)
        self.log("val_seg_dice", loss_dict["seg_dice"], prog_bar=False, sync_dist=True)
        self.log("val_seg_ce", loss_dict["seg_ce"], prog_bar=False, sync_dist=True)
        self.log("val_depth_loss", loss_dict["depth"], prog_bar=True, sync_dist=True)
        self.log("val_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("val_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True)
        self.log("val_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True)
        self.log("val_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        self.log("val_rmse_log", metrics["rmse_log"], prog_bar=False, sync_dist=True)
        self.log("val_delta1", metrics["delta1"], prog_bar=False, sync_dist=True)
        self.log("val_fps", fps, prog_bar=False, sync_dist=True)
        self.log("val_latency", val_latency_ms, prog_bar=False, sync_dist=True)
        
        # Store the first batch for final visualization
        if batch_idx == 0:
            self._last_val_batch = (rgbd.detach().clone(), 
                                   seg_masks.detach().clone(), depth_target.detach().clone())
        
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
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        seg_pred = torch.argmax(torch.softmax(seg_logits.detach(), dim=1), dim=1)
        for c in range(self.num_classes):
            tp = ((seg_pred == c) & (seg_masks == c)).float().sum()
            fp = ((seg_pred == c) & (seg_masks != c)).float().sum()
            fn = ((seg_pred != c) & (seg_masks == c)).float().sum()
            
            self._epoch_val_tp[c] += tp
            self._epoch_val_fp[c] += fp
            self._epoch_val_fn[c] += fn
        
        # Store for epoch curves
        self._epoch_val_metrics.append({
            "loss": float(total_loss.detach().cpu().item()),
            "seg_loss": float(loss_dict["seg"].detach().cpu().item()),
            "seg_dice": float(loss_dict["seg_dice"].detach().cpu().item()),
            "seg_ce": float(loss_dict["seg_ce"].detach().cpu().item()),
            "depth_loss": float(loss_dict["depth"].detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
            "delta1": float(metrics["delta1"].detach().cpu().item()),
        })
        
        return total_loss

    def test_step(self, batch, batch_idx):
        rgbd, seg_masks, depth_target = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                seg_logits, depth_pred = self(rgbd)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            seg_logits, depth_pred = self(rgbd)
            dt = max(1e-9, time.perf_counter() - t0)
        
        total_loss, loss_dict = self._compute_loss(seg_logits, depth_pred, seg_masks, depth_target)
        metrics = self._compute_metrics(seg_logits.detach(), depth_pred.detach(), seg_masks, depth_target)
        
        # FPS and per-image latency (ms)
        batch_size = float(rgbd.shape[0])
        fps = batch_size / float(dt)
        test_latency_ms = (float(dt) / max(1.0, batch_size)) * 1000.0
        
        # Logging
        self.log("test_loss", total_loss, sync_dist=True)
        self.log("test_seg_loss", loss_dict["seg"], sync_dist=True)
        self.log("test_seg_dice", loss_dict["seg_dice"], sync_dist=True)
        self.log("test_seg_ce", loss_dict["seg_ce"], sync_dist=True)
        self.log("test_depth_loss", loss_dict["depth"], sync_dist=True)
        self.log("test_miou", metrics["miou"], sync_dist=True)
        self.log("test_abs_rel", metrics["abs_rel"], sync_dist=True)
        self.log("test_sq_rel", metrics["sq_rel"], sync_dist=True)
        self.log("test_rmse", metrics["rmse"], sync_dist=True)
        self.log("test_rmse_log", metrics["rmse_log"], sync_dist=True)
        self.log("test_delta1", metrics["delta1"], sync_dist=True)
        self.log("test_acc", metrics["seg_acc"], sync_dist=True)
        self.log("test_fps", fps, sync_dist=True)
        self.log("test_latency", test_latency_ms, sync_dist=True)
        
        # Save visuals for test (only first batch to avoid clutter)
        if batch_idx == 0:
            self._maybe_save_visuals(rgbd, seg_masks, depth_target, seg_logits, depth_pred,
                                    stage="test", batch_idx=batch_idx)
        
        return total_loss

    def on_train_epoch_end(self) -> None:
        if self._epoch_train_metrics:
            for key in ["loss", "seg_loss", "seg_dice", "seg_ce", "depth_loss", "miou", "abs_rel", "sq_rel", "rmse"]:
                values = [m[key] for m in self._epoch_train_metrics if key in m]
                if values:
                    curve_key = f"train_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Log epoch-level class-wise IoU
        class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
        for i, class_name in enumerate(class_names):
            if self._epoch_train_tp[i] > 0 or self._epoch_train_fp[i] > 0 or self._epoch_train_fn[i] > 0:
                iou_value = self._epoch_train_tp[i] / (self._epoch_train_tp[i] + self._epoch_train_fp[i] + self._epoch_train_fn[i])
            else:
                iou_value = 0.0
            self.log(f"train_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)
        
        # Reset accumulators
        self._epoch_train_metrics.clear()
        self._epoch_train_tp.zero_()
        self._epoch_train_fp.zero_()
        self._epoch_train_fn.zero_()

    def on_validation_epoch_end(self) -> None:
        if self._epoch_val_metrics:
            for key in ["loss", "seg_loss", "seg_dice", "seg_ce", "depth_loss", "miou", "abs_rel", "sq_rel", "rmse", "delta1"]:
                values = [m[key] for m in self._epoch_val_metrics if key in m]
                if values:
                    curve_key = f"val_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Log epoch-level class-wise IoU
        class_names = ["background", "chamoe", "heatpipe", "path", "pillar", "topdownfarm", "unknown"]
        for i, class_name in enumerate(class_names):
            if self._epoch_val_tp[i] > 0 or self._epoch_val_fp[i] > 0 or self._epoch_val_fn[i] > 0:
                iou_value = self._epoch_val_tp[i] / (self._epoch_val_tp[i] + self._epoch_val_fp[i] + self._epoch_val_fn[i])
            else:
                iou_value = 0.0
            self.log(f"val_class_iou_{class_name}", iou_value, prog_bar=False, sync_dist=True)
        
        # Reset accumulators
        self._epoch_val_metrics.clear()
        self._epoch_val_tp.zero_()
        self._epoch_val_fp.zero_()
        self._epoch_val_fn.zero_()

    def on_fit_end(self) -> None:
        """Save final visualizations and curves at the end of training"""
        try:
            # Save final visualizations from the last validation
            if self.base_vis_dir and self.vis_max > 0:
                if hasattr(self, '_last_val_batch'):
                    rgbd, seg_masks, depth_target = self._last_val_batch
                    with torch.no_grad():
                        seg_logits, depth_pred = self(rgbd)
                        self._maybe_save_visuals(rgbd, seg_masks, depth_target, seg_logits, depth_pred, 
                                                stage="final", batch_idx=0)
            
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
    def _maybe_save_visuals(self, rgbd: torch.Tensor, seg_masks: torch.Tensor, 
                           depth_target: torch.Tensor, seg_logits: torch.Tensor, 
                           depth_pred: torch.Tensor, stage: str, batch_idx: int) -> None:
        if not self.base_vis_dir or self.vis_max <= 0:
            return
        # In test stage, save for all batches; otherwise, only first batch
        if stage != "test" and batch_idx != 0:
            return
        # Save visualization only every 10 epochs (validation only)
        if stage == "val" and self.current_epoch % 10 != 0:
            return
        
        try:
            import os
            from PIL import Image as PILImage
            os.makedirs(os.path.join(self.base_vis_dir, stage), exist_ok=True)

            # Denormalize RGB images (RGBD에서 RGB 부분만 추출)
            rgb_part = rgbd[:, :3]  # RGB 부분만 추출
            imgs = (rgb_part * self.vis_std + self.vis_mean).clamp(0, 1)
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

            save_count = min(self.vis_max, imgs.shape[0])
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
                
                # Use different naming for test stage
                if stage == "test":
                    out_path = os.path.join(self.base_vis_dir, stage, 
                                           f"test_step{self.global_step:06d}_{i}.png")
                else:
                    out_path = os.path.join(self.base_vis_dir, stage, 
                                           f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{i}.png")
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
def build_segformer_datasets(dataset_root: Path, image_size: Tuple[int, int]):
    """Build SegFormer multi-task datasets"""
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "masks"
    train_depth = dataset_root / "train" / "depth"
    
    val_images = dataset_root / "val" / "images"
    val_masks = dataset_root / "val" / "masks"
    val_depth = dataset_root / "val" / "depth"
    
    test_images = dataset_root / "test" / "images"
    test_masks = dataset_root / "test" / "masks"
    test_depth = dataset_root / "test" / "depth"

    mean, std = get_preprocessing_params("segformer")

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
    parser = argparse.ArgumentParser(description="Train SegFormer Multi-Task (Seg + Depth)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--precision", type=str, default="16", 
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
    parser.add_argument("--encoder-name", type=str, default="mit_b2", 
                       choices=["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"],
                       help="SegFormer encoder type")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", 
                       help="Pretrained encoder weights")
    
    return parser.parse_args()


def main() -> None:
    pl.seed_everything(42, workers=True)
    args = parse_args()
    
    # Print CUDA info
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Measure total run time (train + val + test)
    _run_start_time = time.time()

    dataset_root = Path(os.path.abspath(args.dataset_root))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_ds, val_ds, test_ds = build_segformer_datasets(
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
    model = LightningSegformerMTL(
        encoder_name=args.encoder_name,
        num_classes=NUM_CLASSES,
        lr=args.lr,
        scheduler_t_max=args.epochs * steps_per_epoch,
        loss_type=args.loss_type,
        seg_loss_weight=args.seg_loss_weight,
        depth_loss_weight=args.depth_loss_weight,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        save_vis_dir=vis_dir,
        vis_max=args.vis_max,
        save_root_dir=str(output_dir),
        encoder_weights=args.encoder_weights,
    )

    # Callbacks
    ckpt_miou = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="segformer-mtl-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,  # 최고 mIoU만 저장
        save_last=True,
    )
    
    ckpt_absrel = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="segformer-mtl-{epoch:02d}-{val_abs_rel:.4f}",
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

    logger = TensorBoardLogger(save_dir=str(output_dir), name="segformer_mtl_logs")

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


def segformer_main():
    import argparse
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pathlib import Path
    from train_mtl_esanet import LightningMTLSegformer, build_mtl_datasets, NUM_CLASSES
    import os
    import torch

    parser = argparse.ArgumentParser(description="Train SegFormer MTL (Seg + Depth)")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="mit_b2")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--loss-type", type=str, default="silog", choices=["silog", "l1"]) 
    parser.add_argument("--vis-dir", type=str, default="")
    parser.add_argument("--vis-max", type=int, default=4)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)

    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    dataset_root = Path(os.path.abspath(args.dataset_root))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_mtl_datasets(
        dataset_root, args.encoder, (args.image_size[0], args.image_size[1])
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    steps_per_epoch = max(1, len(train_loader))

    vis_dir = args.vis_dir
    if vis_dir.strip().lower() == "none":
        vis_dir = ""
    elif vis_dir.strip() == "":
        vis_dir = str(output_dir / "vis")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    model = LightningMTLSegformer(
        encoder_name=args.encoder,
        num_classes=NUM_CLASSES,
        lr=args.lr,
        scheduler_t_max=args.epochs * steps_per_epoch,
        loss_type=args.loss_type,
        save_vis_dir=vis_dir,
        vis_max=args.vis_max,
        save_root_dir=str(output_dir),
    )

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="mtl-segformer-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_abs_rel",
        min_delta=0.001,
        patience=30,
        verbose=True,
        mode="min",
    )
    logger = TensorBoardLogger(save_dir=str(output_dir), name="mtl_logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        default_root_dir=str(output_dir),
        callbacks=[ckpt, early_stop],
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=(args.ckpt_path or None))
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt.best_model_path)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)

    summary = {
        "best_checkpoint": ckpt.best_model_path,
        "best_miou": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        "best_val_results": best_val_results,
        "test_results": test_results,
    }
    print("=" * 80)
    print("SegFormer MTL Training Complete!")
    print("=" * 80)
    print(summary)


if __name__ == "__main__":
    main()
