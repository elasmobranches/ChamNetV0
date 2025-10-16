"""
Inference script for Multi-Task SegFormer (Segmentation + Depth)
"""
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from train_mtl_segformer import LightningMTLSegformer, ID2LABEL


def load_model(checkpoint_path: str, device: str = "cuda") -> LightningMTLSegformer:
    """Load model from checkpoint"""
    model = LightningMTLSegformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def preprocess_image(
    image_path: str,
    encoder_name: str,
    image_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    
    # Get preprocessing params
    try:
        params = smp.encoders.get_preprocessing_params(encoder_name)
        mean = params["mean"]
        std = params["std"]
    except:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Transform
    image = TF.resize(image, [image_size[1], image_size[0]], interpolation=T.InterpolationMode.BILINEAR)
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(image_tensor, mean=mean, std=std)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original


@torch.no_grad()
def inference(
    model: LightningMTLSegformer,
    image_tensor: torch.Tensor,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Run inference with accurate GPU timing"""
    image_tensor = image_tensor.to(device)
    
    # Measure inference time (GPU-accurate timing)
    if torch.cuda.is_available() and device == "cuda":
        # Create CUDA events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start event
        start_event.record()
        
        # Forward pass
        seg_logits, depth_pred = model(image_tensor)
        
        # Record end event
        end_event.record()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Calculate elapsed time in milliseconds
        inference_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    else:
        # Fallback to CPU timing
        import time
        t0 = time.perf_counter()
        seg_logits, depth_pred = model(image_tensor)
        inference_time = time.perf_counter() - t0
    
    # Get segmentation prediction
    seg_pred = torch.argmax(seg_logits, dim=1).cpu()
    
    # Get depth prediction
    depth_pred = depth_pred.squeeze(1).cpu()
    
    return seg_pred, depth_pred, inference_time


def colorize_segmentation(seg_pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Colorize segmentation prediction"""
    h, w = seg_pred.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(palette.shape[0]):
        colored[seg_pred == c] = palette[c]
    return colored


def colorize_depth(depth: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Colorize depth map"""
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_norm)[:, :, :3]
    return (depth_colored * 255).astype(np.uint8)


def visualize_results(
    original: np.ndarray,
    seg_pred: np.ndarray,
    depth_pred: np.ndarray,
    palette: np.ndarray,
    save_path: str = None
) -> None:
    """Visualize and save results"""
    # Resize predictions to original size
    h, w = original.shape[:2]
    seg_pred_resized = np.array(Image.fromarray(seg_pred.astype(np.uint8)).resize((w, h), Image.NEAREST))
    depth_pred_resized = np.array(Image.fromarray(depth_pred).resize((w, h), Image.BILINEAR))
    
    # Colorize
    seg_colored = colorize_segmentation(seg_pred_resized, palette)
    depth_colored = colorize_depth(depth_pred_resized)
    
    # Create panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=14)
    axes[0, 0].axis("off")
    
    # Segmentation prediction
    axes[0, 1].imshow(seg_colored)
    axes[0, 1].set_title("Segmentation Prediction", fontsize=14)
    axes[0, 1].axis("off")
    
    # Depth prediction (colored)
    axes[1, 0].imshow(depth_colored)
    axes[1, 0].set_title("Depth Prediction (Colored)", fontsize=14)
    axes[1, 0].axis("off")
    
    # Depth prediction (grayscale)
    axes[1, 1].imshow(depth_pred_resized, cmap="gray")
    axes[1, 1].set_title("Depth Prediction (Grayscale)", fontsize=14)
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Results saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_legend(palette: np.ndarray, id2label: dict, save_path: str = None) -> None:
    """Create segmentation legend"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = []
    for i, (label_id, label_name) in enumerate(id2label.items()):
        if i < len(palette):
            color = palette[i] / 255.0
            legend_elements.append(Patch(facecolor=color, label=label_name))
    
    ax.legend(handles=legend_elements, loc="center", fontsize=12, frameon=True)
    ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Legend saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inference for MTL SegFormer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="./inference_results", 
                       help="Output directory")
    parser.add_argument("--encoder", type=str, default="mit_b2", 
                       help="Encoder name (must match checkpoint)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                       help="Image size (width height)")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--save-legend", action="store_true", 
                       help="Save segmentation legend")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Device: {args.device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully")
    
    # Preprocess image
    print(f"Preprocessing image {args.image}...")
    image_tensor, original = preprocess_image(
        args.image, 
        args.encoder, 
        (args.image_size[0], args.image_size[1])
    )
    
    # Inference
    print("Running inference...")
    seg_pred, depth_pred, inference_time = inference(model, image_tensor, device=args.device)
    
    # Convert to numpy
    seg_pred = seg_pred.squeeze(0).numpy()
    depth_pred = depth_pred.squeeze(0).numpy()
    
    print(f"Segmentation shape: {seg_pred.shape}")
    print(f"Depth shape: {depth_pred.shape}")
    print(f"Unique seg classes: {np.unique(seg_pred)}")
    print(f"Depth range: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"FPS: {1.0 / inference_time:.2f}")
    print(f"Latency: {inference_time*1000:.2f} ms")
    
    # Get palette
    palette = model.palette.cpu().numpy()
    
    # Visualize
    image_name = Path(args.image).stem
    output_path = output_dir / f"{image_name}_result.png"
    
    print("Visualizing results...")
    visualize_results(original, seg_pred, depth_pred, palette, save_path=str(output_path))
    
    # Save legend if requested
    if args.save_legend:
        legend_path = output_dir / "segmentation_legend.png"
        create_legend(palette, ID2LABEL, save_path=str(legend_path))
    
    # Save individual results
    seg_colored = colorize_segmentation(seg_pred, palette)
    depth_colored = colorize_depth(depth_pred)
    
    Image.fromarray(seg_colored).save(output_dir / f"{image_name}_seg.png")
    Image.fromarray(depth_colored).save(output_dir / f"{image_name}_depth_colored.png")
    Image.fromarray((depth_pred * 255).astype(np.uint8)).save(output_dir / f"{image_name}_depth_gray.png")
    
    print(f"Inference complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

