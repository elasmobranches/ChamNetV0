"""
Depth Estimation Only Training Script
- Baseline for comparison with Multi-Task Learning
"""
import argparse
import os
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
# Depth Dataset
# ============================================================================
class DepthDataset(Dataset):
    """Depth estimation only dataset"""
    def __init__(
        self,
        images_dir: Path,
        depth_dir: Path,
        image_size: Tuple[int, int],
        mean: List[float],
        std: List[float],
        is_train: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.depth_dir = depth_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train

        self.image_files: List[str] = [
            f for f in sorted(os.listdir(images_dir)) if os.path.isfile(images_dir / f)
        ]
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        # Depth path: {stem}_depth.png
        depth_name = f"{Path(img_name).stem}_depth.png"
        depth_path = self.depth_dir / depth_name

        # Load data as PIL Images
        image = Image.open(img_path).convert("RGB")
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
        depth = TF.resize(depth, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)

        # Optional augmentation
        # if self.is_train:
        #     if torch.rand(1) > 0.5:
        #         image = TF.hflip(image)
        #         depth = TF.hflip(depth)

        # Convert to tensors
        image = TF.to_tensor(image)
        depth = torch.from_numpy(np.array(depth, dtype=np.float32))

        # Normalize image
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, depth


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    """Get preprocessing parameters for encoder"""
    try:
        params = smp.encoders.get_preprocessing_params(encoder_name)
        return params["mean"], params["std"]
    except:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================
# Loss Functions
# ============================================================================
class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss"""
    def __init__(self, lambda_variance: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.lambda_variance = lambda_variance
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
        
        log_diff = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        loss = (log_diff ** 2).mean() - self.lambda_variance * (log_diff.mean() ** 2)
        return loss


class L1DepthLoss(nn.Module):
    """L1 loss for depth estimation"""
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
# Depth Metrics
# ============================================================================
@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Compute standard depth estimation metrics"""
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
    
    abs_rel = (torch.abs(pred - target) / (target + eps)).mean()
    sq_rel = ((pred - target) ** 2 / (target + eps)).mean()
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(pred + eps) - torch.log(target + eps)) ** 2).mean())
    
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
# Depth Model
# ============================================================================
class DepthSegformer(nn.Module):
    """SegFormer for depth estimation only"""
    def __init__(
        self,
        encoder_name: str = "mit_b0",
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()
        
        self.model = smp.create_model(
            arch="Segformer",
            encoder_name=encoder_name,
            in_channels=3,
            classes=1,  # Single channel for depth
            encoder_weights=encoder_weights,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth_raw = self.model(x)
        depth_pred = torch.sigmoid(depth_raw)
        return depth_pred


# ============================================================================
# PyTorch Lightning Module
# ============================================================================
class LightningDepthSegformer(pl.LightningModule):
    """PyTorch Lightning module for Depth Estimation"""
    def __init__(
        self,
        encoder_name: str,
        lr: float,
        scheduler_t_max: int,
        loss_type: str = "silog",
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DepthSegformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
        )
        
        if loss_type == "silog":
            self.loss_fn = SILogLoss()
        else:
            self.loss_fn = L1DepthLoss()
        
        self.lr = lr
        self.t_max = scheduler_t_max
        self.base_vis_dir = save_vis_dir
        self.vis_max = int(vis_max) if vis_max is not None else 0
        self.save_root_dir = save_root_dir
        
        # Curves storage
        self.curves = {
            "train_loss": [],
            "val_loss": [],
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
        
        # Store normalization params
        params = smp.encoders.get_preprocessing_params(encoder_name)
        mean = params["mean"]
        std = params["std"]
        self.register_buffer("vis_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("vis_std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, depth_maps = batch
        depth_pred = self(images)
        
        loss = self.loss_fn(depth_pred, depth_maps)
        metrics = compute_depth_metrics(depth_pred.detach(), depth_maps)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True)
        self.log("train_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True)
        self.log("train_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        self.log("train_delta1", metrics["delta1"], prog_bar=False, sync_dist=True)
        
        self._epoch_train_metrics.append({
            "loss": float(loss.detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, depth_maps = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                depth_pred = self(images)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            depth_pred = self(images)
            dt = max(1e-9, time.perf_counter() - t0)
        
        loss = self.loss_fn(depth_pred, depth_maps)
        metrics = compute_depth_metrics(depth_pred.detach(), depth_maps)
        
        fps = float(images.shape[0]) / float(dt)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_abs_rel", metrics["abs_rel"], prog_bar=True, sync_dist=True)
        self.log("val_sq_rel", metrics["sq_rel"], prog_bar=False, sync_dist=True)
        self.log("val_rmse", metrics["rmse"], prog_bar=False, sync_dist=True)
        self.log("val_delta1", metrics["delta1"], prog_bar=False, sync_dist=True)
        self.log("val_fps", fps, prog_bar=False, sync_dist=True)
        
        self._maybe_save_visuals(images, depth_maps, depth_pred, stage="val", batch_idx=batch_idx)
        
        self._epoch_val_metrics.append({
            "loss": float(loss.detach().cpu().item()),
            "abs_rel": float(metrics["abs_rel"].detach().cpu().item()),
            "sq_rel": float(metrics["sq_rel"].detach().cpu().item()),
            "rmse": float(metrics["rmse"].detach().cpu().item()),
            "delta1": float(metrics["delta1"].detach().cpu().item()),
        })
        
        return loss

    def test_step(self, batch, batch_idx):
        images, depth_maps = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                depth_pred = self(images)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            depth_pred = self(images)
            dt = max(1e-9, time.perf_counter() - t0)
        
        loss = self.loss_fn(depth_pred, depth_maps)
        metrics = compute_depth_metrics(depth_pred.detach(), depth_maps)
        
        fps = float(images.shape[0]) / float(dt)
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_abs_rel", metrics["abs_rel"], sync_dist=True)
        self.log("test_sq_rel", metrics["sq_rel"], sync_dist=True)
        self.log("test_rmse", metrics["rmse"], sync_dist=True)
        self.log("test_delta1", metrics["delta1"], sync_dist=True)
        self.log("test_fps", fps, sync_dist=True)
        
        self._maybe_save_visuals(images, depth_maps, depth_pred, stage="test", batch_idx=batch_idx)
        
        return loss

    def on_train_epoch_end(self) -> None:
        if self._epoch_train_metrics:
            for key in ["loss", "abs_rel", "sq_rel", "rmse"]:
                values = [m[key] for m in self._epoch_train_metrics if key in m]
                if values:
                    curve_key = f"train_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        self._epoch_train_metrics.clear()

    def on_validation_epoch_end(self) -> None:
        if self._epoch_val_metrics:
            for key in ["loss", "abs_rel", "sq_rel", "rmse", "delta1"]:
                values = [m[key] for m in self._epoch_val_metrics if key in m]
                if values:
                    curve_key = f"val_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        self._epoch_val_metrics.clear()

    @torch.no_grad()
    def _maybe_save_visuals(self, images: torch.Tensor, depth_maps: torch.Tensor,
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

            imgs = (images * self.vis_std + self.vis_mean).clamp(0, 1)
            imgs = (imgs * 255.0).to(torch.uint8).cpu()

            depth_preds = depth_pred.squeeze(1).cpu()
            depth_gts = depth_maps.cpu()
            
            def colorize_depth(depth_hw: torch.Tensor) -> np.ndarray:
                depth_np = depth_hw.numpy()
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
                return (depth_colored * 255).astype(np.uint8)

            save_count = min(self.vis_max, imgs.shape[0])
            for i in range(save_count):
                img = imgs[i].permute(1, 2, 0).numpy()
                depth_gt = colorize_depth(depth_gts[i])
                depth_pr = colorize_depth(depth_preds[i])
                
                # Standardized format: [GT, Original, Prediction]
                panel = np.concatenate([depth_gt, img, depth_pr], axis=1)
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

    def on_fit_end(self) -> None:
        try:
            if not self.save_root_dir:
                return
            curves_dir = os.path.join(self.save_root_dir, "curves")
            os.makedirs(curves_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss
            ax = axes[0, 0]
            if len(self.curves["train_loss"]) > 0:
                ax.plot(self.curves["train_loss"], label="train")
            if len(self.curves["val_loss"]) > 0:
                ax.plot(self.curves["val_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            
            # AbsRel
            ax = axes[0, 1]
            if len(self.curves["train_abs_rel"]) > 0:
                ax.plot(self.curves["train_abs_rel"], label="train")
            if len(self.curves["val_abs_rel"]) > 0:
                ax.plot(self.curves["val_abs_rel"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("AbsRel (lower is better)")
            ax.grid(True)
            ax.legend()
            
            # RMSE
            ax = axes[1, 0]
            if len(self.curves["train_rmse"]) > 0:
                ax.plot(self.curves["train_rmse"], label="train")
            if len(self.curves["val_rmse"]) > 0:
                ax.plot(self.curves["val_rmse"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("RMSE (lower is better)")
            ax.grid(True)
            ax.legend()
            
            # δ<1.25
            ax = axes[1, 1]
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
def build_depth_datasets(dataset_root: Path, encoder_name: str, image_size: Tuple[int, int]):
    """Build depth datasets"""
    train_images = dataset_root / "train" / "images"
    train_depth = dataset_root / "train" / "depth"
    
    val_images = dataset_root / "val" / "images"
    val_depth = dataset_root / "val" / "depth"
    
    test_images = dataset_root / "test" / "images"
    test_depth = dataset_root / "test" / "depth"

    mean, std = get_preprocessing_params(encoder_name)

    train_ds = DepthDataset(train_images, train_depth, image_size=image_size, mean=mean, std=std, is_train=True)
    val_ds = DepthDataset(val_images, val_depth, image_size=image_size, mean=mean, std=std, is_train=False)
    test_ds = DepthDataset(test_images, test_depth, image_size=image_size, mean=mean, std=std, is_train=False)
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Depth-Only SegFormer")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="mit_b0")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--loss-type", type=str, default="silog", choices=["silog", "l1"])
    parser.add_argument("--vis-dir", type=str, default="")
    parser.add_argument("--vis-max", type=int, default=4)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    
    return parser.parse_args()


def main() -> None:
    pl.seed_everything(42, workers=True)
    args = parse_args()
    
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

    train_ds, val_ds, test_ds = build_depth_datasets(
        dataset_root, args.encoder, (args.image_size[0], args.image_size[1])
    )
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    steps_per_epoch = max(1, len(train_loader))
    
    vis_dir = args.vis_dir
    if vis_dir.strip().lower() == "none":
        vis_dir = ""
    elif vis_dir.strip() == "":
        vis_dir = str(output_dir / "vis")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    model = LightningDepthSegformer(
        encoder_name=args.encoder,
        lr=args.lr,
        scheduler_t_max=args.epochs * steps_per_epoch,
        loss_type=args.loss_type,
        save_vis_dir=vis_dir,
        vis_max=args.vis_max,
        save_root_dir=str(output_dir),
    )

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="depth-segformer-{epoch:02d}-{val_abs_rel:.4f}",
        monitor="val_abs_rel",
        mode="min",
        save_top_k=1,  # 최소 AbsRel만 저장
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_abs_rel",
        min_delta=0.001,
        patience=30,  # Increased for stable convergence
        verbose=True,
        mode="min",
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="depth_logs")

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
    
    print("=" * 80)
    print("Starting Depth-Only training...")
    print("=" * 80)
    trainer.fit(model, train_loader, val_loader, ckpt_path=(args.ckpt_path or None))
    
    print("=" * 80)
    print("Validating with best checkpoint...")
    print("=" * 80)
    best_val_results = trainer.validate(dataloaders=val_loader, ckpt_path=ckpt.best_model_path)
    
    print("=" * 80)
    print("Testing...")
    print("=" * 80)
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)

    # Compute elapsed time
    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)
    
    summary = {
        "best_checkpoint": ckpt.best_model_path,
        "best_abs_rel": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
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

