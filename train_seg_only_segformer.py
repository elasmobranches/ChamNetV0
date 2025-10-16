"""
Segmentation Only Training Script
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
# Constants
# ============================================================================
NUM_CLASSES = 7  # 원래 정의된 클래스 수 유지
ID2LABEL: Dict[int, str] = {
    0: "background",
    1: "chamoe",
    2: "heatpipe", 
    3: "path",
    4: "pillar",      # 데이터에 없지만 정의는 유지
    5: "topdownfarm",
    6: "unknown",
}


# ============================================================================
# Segmentation Dataset
# ============================================================================
class SegmentationDataset(Dataset):
    """Segmentation only dataset"""
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
            f for f in sorted(os.listdir(images_dir)) if os.path.isfile(images_dir / f)
        ]
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        
        # Mask path: {stem}_mask.png
        mask_name = f"{Path(img_name).stem}_mask.png"
        mask_path = self.masks_dir / mask_name

        # Load data as PIL Images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = TF.resize(image, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size[1], self.image_size[0]], interpolation=T.InterpolationMode.NEAREST)

        # Optional augmentation
        # if self.is_train:
        #     if torch.rand(1) > 0.5:
        #         image = TF.hflip(image)
        #         mask = TF.hflip(mask)

        # Convert to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # Normalize image
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, mask


def get_preprocessing_params(encoder_name: str) -> Tuple[List[float], List[float]]:
    """Get preprocessing parameters for encoder"""
    try:
        params = smp.encoders.get_preprocessing_params(encoder_name)
        return params["mean"], params["std"]
    except:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============================================================================
# Segmentation Model
# ============================================================================
class SegmentationSegformer(nn.Module):
    """SegFormer for segmentation only"""
    def __init__(
        self,
        encoder_name: str = "mit_b0",
        num_classes: int = 7,
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()
        
        self.model = smp.create_model(
            arch="Segformer",
            encoder_name=encoder_name,
            in_channels=3,
            classes=num_classes,
            encoder_weights=encoder_weights,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# PyTorch Lightning Module
# ============================================================================
class LightningSegSegformer(pl.LightningModule):
    """PyTorch Lightning module for Segmentation"""
    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        lr: float,
        scheduler_t_max: int,
        save_vis_dir: str = "",
        vis_max: int = 4,
        save_root_dir: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SegmentationSegformer(
            encoder_name=encoder_name,
            num_classes=num_classes,
            encoder_weights="imagenet",
        )
        
        # Combined Dice + Cross-Entropy for better class balance
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()
        
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
            "train_dice": [],
            "val_dice": [],
            "train_ce": [],
            "val_ce": [],
            "train_miou": [],
            "val_miou": [],
            "train_acc": [],
            "val_acc": [],
        }
        self._epoch_train_metrics = []
        self._epoch_val_metrics = []
        
        # Class-wise IoU accumulation for epoch-level calculation
        self._epoch_train_tp = torch.zeros(self.num_classes)
        self._epoch_train_fp = torch.zeros(self.num_classes)
        self._epoch_train_fn = torch.zeros(self.num_classes)
        self._epoch_val_tp = torch.zeros(self.num_classes)
        self._epoch_val_fp = torch.zeros(self.num_classes)
        self._epoch_val_fn = torch.zeros(self.num_classes)
        
        # Store normalization params
        params = smp.encoders.get_preprocessing_params(encoder_name)
        mean = params["mean"]
        std = params["std"]
        self.register_buffer("vis_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("vis_std", torch.tensor(std).view(1, 3, 1, 1))
        
        # Color palette
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @torch.no_grad()
    def _compute_metrics(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, target, mode="multiclass", num_classes=self.num_classes
        )
        miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        rec = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        prec = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        
        # Class-wise IoU
        class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        
        return {
            "miou": miou,
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "class_iou": class_iou,
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        
        # Combined Dice + Cross-Entropy loss
        dice = self.dice_loss(logits, masks)
        ce = self.ce_loss(logits, masks)
        loss = 0.7 * dice + 0.3 * ce  # Weighted combination
        
        metrics = self._compute_metrics(logits.detach(), masks)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_dice", dice, prog_bar=False, sync_dist=True)
        self.log("train_ce", ce, prog_bar=False, sync_dist=True)
        self.log("train_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("train_acc", metrics["accuracy"], prog_bar=False, sync_dist=True)
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        tp, fp, fn, tn = smp.metrics.get_stats(
            torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1), 
            masks, mode="multiclass", num_classes=self.num_classes
        )
        batch_tp = tp.sum(dim=0)  # Sum across batch dimension
        batch_fp = fp.sum(dim=0)
        batch_fn = fn.sum(dim=0)
        
        self._epoch_train_tp += batch_tp
        self._epoch_train_fp += batch_fp
        self._epoch_train_fn += batch_fn
        
        self._epoch_train_metrics.append({
            "loss": float(loss.detach().cpu().item()),
            "dice": float(dice.detach().cpu().item()),
            "ce": float(ce.detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "acc": float(metrics["accuracy"].detach().cpu().item()),
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                logits = self(images)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            logits = self(images)
            dt = max(1e-9, time.perf_counter() - t0)
        
        # Combined Dice + Cross-Entropy loss
        dice = self.dice_loss(logits, masks)
        ce = self.ce_loss(logits, masks)
        loss = 0.7 * dice + 0.3 * ce  # Weighted combination
        
        metrics = self._compute_metrics(logits.detach(), masks)
        
        fps = float(images.shape[0]) / float(dt)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dice, prog_bar=False, sync_dist=True)
        self.log("val_ce", ce, prog_bar=False, sync_dist=True)
        self.log("val_miou", metrics["miou"], prog_bar=True, sync_dist=True)
        self.log("val_acc", metrics["accuracy"], prog_bar=False, sync_dist=True)
        self.log("val_fps", fps, prog_bar=False, sync_dist=True)
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        tp, fp, fn, tn = smp.metrics.get_stats(
            torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1), 
            masks, mode="multiclass", num_classes=self.num_classes
        )
        batch_tp = tp.sum(dim=0)  # Sum across batch dimension
        batch_fp = fp.sum(dim=0)
        batch_fn = fn.sum(dim=0)
        
        self._epoch_val_tp += batch_tp
        self._epoch_val_fp += batch_fp
        self._epoch_val_fn += batch_fn
        
        self._maybe_save_visuals(images, masks, logits, stage="val", batch_idx=batch_idx)
        
        self._epoch_val_metrics.append({
            "loss": float(loss.detach().cpu().item()),
            "dice": float(dice.detach().cpu().item()),
            "ce": float(ce.detach().cpu().item()),
            "miou": float(metrics["miou"].detach().cpu().item()),
            "acc": float(metrics["accuracy"].detach().cpu().item()),
        })
        
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        # Measure inference time (GPU-accurate timing)
        if torch.cuda.is_available():
            # Create CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass with no_grad for inference
            with torch.no_grad():
                logits = self(images)
            
            # Record end event
            end_event.record()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            dt = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            # Fallback to CPU timing
            t0 = time.perf_counter()
            logits = self(images)
            dt = max(1e-9, time.perf_counter() - t0)
        
        # Combined Dice + Cross-Entropy loss
        dice = self.dice_loss(logits, masks)
        ce = self.ce_loss(logits, masks)
        loss = 0.7 * dice + 0.3 * ce  # Weighted combination
        
        metrics = self._compute_metrics(logits.detach(), masks)
        
        fps = float(images.shape[0]) / float(dt)
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_dice", dice, sync_dist=True)
        self.log("test_ce", ce, sync_dist=True)
        self.log("test_miou", metrics["miou"], sync_dist=True)
        self.log("test_acc", metrics["accuracy"], sync_dist=True)
        self.log("test_fps", fps, sync_dist=True)
        
        # Accumulate class-wise statistics for epoch-level IoU calculation
        tp, fp, fn, tn = smp.metrics.get_stats(
            torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1), 
            masks, mode="multiclass", num_classes=self.num_classes
        )
        batch_tp = tp.sum(dim=0)  # Sum across batch dimension
        batch_fp = fp.sum(dim=0)
        batch_fn = fn.sum(dim=0)
        
        # Store for epoch-end calculation (we'll use validation accumulators for test too)
        self._epoch_val_tp += batch_tp
        self._epoch_val_fp += batch_fp
        self._epoch_val_fn += batch_fn
        
        self._maybe_save_visuals(images, masks, logits, stage="test", batch_idx=batch_idx)
        
        return loss

    def on_train_epoch_end(self) -> None:
        if self._epoch_train_metrics:
            for key in ["loss", "dice", "ce", "miou", "acc"]:
                values = [m[key] for m in self._epoch_train_metrics if key in m]
                if values:
                    curve_key = f"train_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Log epoch-level class-wise IoU
        for i, class_name in enumerate(ID2LABEL.values()):
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
            for key in ["loss", "dice", "ce", "miou", "acc"]:
                values = [m[key] for m in self._epoch_val_metrics if key in m]
                if values:
                    curve_key = f"val_{key}"
                    if curve_key in self.curves:
                        self.curves[curve_key].append(float(np.mean(values)))
        
        # Log epoch-level class-wise IoU
        for i, class_name in enumerate(ID2LABEL.values()):
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

    @torch.no_grad()
    def _maybe_save_visuals(self, images: torch.Tensor, masks: torch.Tensor,
                           logits: torch.Tensor, stage: str, batch_idx: int) -> None:
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

            preds = torch.softmax(logits, dim=1).argmax(dim=1).to(torch.int64).cpu()
            gts = masks.to(torch.int64).cpu()
            
            pal = self.palette.cpu()
            
            def colorize(label_hw: torch.Tensor) -> np.ndarray:
                h, w = label_hw.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(min(self.num_classes, pal.shape[0])):
                    out[label_hw.numpy() == c] = pal[c].numpy()
                return out

            save_count = min(self.vis_max, imgs.shape[0])
            for i in range(save_count):
                img = imgs[i].permute(1, 2, 0).numpy()
                gt = colorize(gts[i])
                pr = colorize(preds[i])
                
                # Standardized format: [GT, Original, Prediction]
                panel = np.concatenate([gt, img, pr], axis=1)
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

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss
            ax = axes[0]
            if len(self.curves["train_loss"]) > 0:
                ax.plot(self.curves["train_loss"], label="train")
            if len(self.curves["val_loss"]) > 0:
                ax.plot(self.curves["val_loss"], label="val")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            ax.set_title("Loss Curves")
            
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
            ax.set_title("mIoU Curves")
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "seg_metrics.png"))
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
def build_seg_datasets(dataset_root: Path, encoder_name: str, image_size: Tuple[int, int]):
    """Build segmentation datasets"""
    train_images = dataset_root / "train" / "images"
    train_masks = dataset_root / "train" / "masks"
    
    val_images = dataset_root / "val" / "images"
    val_masks = dataset_root / "val" / "masks"
    
    test_images = dataset_root / "test" / "images"
    test_masks = dataset_root / "test" / "masks"

    mean, std = get_preprocessing_params(encoder_name)

    train_ds = SegmentationDataset(train_images, train_masks, image_size=image_size, mean=mean, std=std, is_train=True)
    val_ds = SegmentationDataset(val_images, val_masks, image_size=image_size, mean=mean, std=std, is_train=False)
    test_ds = SegmentationDataset(test_images, test_masks, image_size=image_size, mean=mean, std=std, is_train=False)
    
    return train_ds, val_ds, test_ds


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Segmentation-Only SegFormer")
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
    
    # Measure total run time (train + val + test)
    _run_start_time = time.time()

    dataset_root = Path(os.path.abspath(args.dataset_root))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_seg_datasets(
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

    model = LightningSegSegformer(
        encoder_name=args.encoder,
        num_classes=NUM_CLASSES,
        lr=args.lr,
        scheduler_t_max=args.epochs * steps_per_epoch,
        save_vis_dir=vis_dir,
        vis_max=args.vis_max,
        save_root_dir=str(output_dir),
    )

    ckpt = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="seg-segformer-{epoch:02d}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_top_k=1,  # 최고 mIoU만 저장
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_miou",
        min_delta=0.001,
        patience=30,  # Increased for stable convergence
        verbose=True,
        mode="max",
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="seg_logs")

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
    print("Starting Segmentation-Only training...")
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
    
    summary = {
        "best_checkpoint": ckpt.best_model_path,
        "best_miou": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        "best_val_results": best_val_results,
        "test_results": test_results,
    }
    
    print("=" * 80)
    # Compute elapsed time
    _elapsed_sec = max(0.0, time.time() - _run_start_time)
    _elapsed_h = int(_elapsed_sec // 3600)
    _elapsed_m = int((_elapsed_sec % 3600) // 60)
    _elapsed_s = int(_elapsed_sec % 60)

    print("학습완료!")
    print("=" * 80)
    print(summary)
    print(f"⏱️ 몇분이나 걸렸을까요?: {_elapsed_h:02d}:{_elapsed_m:02d}:{_elapsed_s:02d} ({_elapsed_sec:.2f}s)")


if __name__ == "__main__":
    main()

