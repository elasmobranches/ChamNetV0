#!/usr/bin/env python3
"""
Experiment Runner with Version Management
각 실험을 버전별로 관리하여 혼란 방지
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def create_experiment_dir(base_dir: str, exp_name: str) -> Path:
    """Create versioned experiment directory"""
    exp_dir = Path(base_dir) / f"{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir: Path, config: dict):
    """Save experiment configuration"""
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_file}")


def run_training(script: str, exp_dir: Path, args: dict):
    """Run training script with arguments"""
    cmd = [sys.executable, script]
    
    # Add arguments
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            for item in value:
                cmd.append(str(item))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Redirect output
    log_file = exp_dir / "training.log"
    
    print(f"\n🚀 Starting training...")
    print(f"   Script: {script}")
    print(f"   Log: {log_file}")
    print(f"   Command: {' '.join(cmd)}")
    print("")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to both file and terminal
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    return process.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run versioned experiment")
    
    # Experiment type
    parser.add_argument("--exp-type", type=str, 
                       choices=["seg", "depth", "mtl", "esanet_mtl"],
                       help="Experiment type", 
                       default="mtl")
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Custom experiment name (default: auto)")
    parser.add_argument("--base-dir", type=str, default="./experiments",
                       help="Base directory for experiments")
    
    # Common training args
    parser.add_argument("--dataset-root", type=str, default="../dataset")
    parser.add_argument("--encoder", type=str, default="mit_b2")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="16")
    
    # MTL-specific
    parser.add_argument("--use-uncertainty-weighting", action="store_true")
    parser.add_argument("--loss-type", type=str, default="silog")
    
    # ESANet-specific
    parser.add_argument("--height", type=int, default=512, help="Image height for ESANet")
    parser.add_argument("--width", type=int, default=512, help="Image width for ESANet")
    parser.add_argument("--encoder-rgb", type=str, default="resnet34", help="RGB encoder for ESANet")
    parser.add_argument("--encoder-depth", type=str, default="resnet34", help="Depth encoder for ESANet")
    parser.add_argument("--encoder-block", type=str, default="NonBottleneck1D", help="Encoder block for ESANet")
    parser.add_argument("--pretrained-path", type=str, 
                       default="/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth",
                       help="Path to pretrained ESANet weights")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        if args.exp_type == "esanet_mtl":
            # ESANet uses encoder-rgb instead of encoder
            exp_name = f"{args.exp_type}_{args.encoder_rgb}"
        else:
            exp_name = f"{args.exp_type}_{args.encoder}"
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.base_dir, exp_name)
    
    print("=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"📁 Directory: {exp_dir}")
    print(f"🔬 Type: {args.exp_type}")
    print("=" * 80)
    
    # Prepare training arguments
    train_args = {
        "dataset-root": args.dataset_root,
        "output-dir": str(exp_dir),
        "encoder": args.encoder,
        "image-size": args.image_size,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "num-workers": 4,
        "precision": args.precision,
        "accelerator": "gpu",
        "devices": "1",
        "gradient-clip-val": 1.0,
        "vis-max": 4,
    }
    
    # Select training script
    if args.exp_type == "seg":
        script = "train_seg_only_segformer.py"
    elif args.exp_type == "depth":
        script = "train_depth_only_segformer.py"
        train_args["loss-type"] = args.loss_type
    elif args.exp_type == "mtl":
        # 반드시 SegFormer용 트레이너를 호출
        script = "train_mtl_segformer.py"
        train_args["loss-type"] = args.loss_type
        if args.use_uncertainty_weighting:
            train_args["use-uncertainty-weighting"] = True

        # train_mtl_segformer.py는 SegFormer 전용 인자(encoder-name, height, width)를 받으므로 수정
        # encoder -> encoder-name으로 변경
        if "encoder" in train_args:
            train_args["encoder-name"] = train_args.pop("encoder")
        # image-size -> height, width로 변경
        if "image-size" in train_args:
            image_size = train_args.pop("image-size")
            train_args["height"] = image_size[0]
            train_args["width"] = image_size[1]
    elif args.exp_type == "esanet_mtl":
        script = "train_mtl_esanet.py"
        train_args["loss-type"] = args.loss_type
        if args.use_uncertainty_weighting:
            train_args["use-uncertainty-weighting"] = True
        # ESANet-specific arguments
        train_args["height"] = args.height
        train_args["width"] = args.width
        train_args["encoder-rgb"] = args.encoder_rgb
        train_args["encoder-depth"] = args.encoder_depth
        train_args["encoder-block"] = args.encoder_block
        train_args["pretrained-path"] = args.pretrained_path
        # Remove incompatible arguments for ESANet
        if "encoder" in train_args:
            del train_args["encoder"]
        if "image-size" in train_args:
            del train_args["image-size"]
    
    # Save configuration
    config = {
        "experiment_name": exp_name,
        "experiment_type": args.exp_type,
        "timestamp": datetime.now().isoformat(),
        "script": script,
        "arguments": train_args,
    }
    save_config(exp_dir, config)
    
    # Run training
    returncode = run_training(script, exp_dir, train_args)
    
    if returncode == 0:
        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📁 Results: {exp_dir}")
        print(f"📊 Checkpoints: {exp_dir}/*.ckpt")
        print(f"📈 Curves: {exp_dir}/curves/")
        print(f"👁️  Visualizations: {exp_dir}/vis/")
        print(f"📝 Log: {exp_dir}/training.log")
        print(f"⚙️  Config: {exp_dir}/config.json")
    else:
        print("\n" + "=" * 80)
        print("❌ EXPERIMENT FAILED")
        print("=" * 80)
        print(f"Check log: {exp_dir}/training.log")
    
    return returncode


if __name__ == "__main__":
    sys.exit(main())

