#!/usr/bin/env python3
"""
Experiment Runner for ESANet MTL (separate from SegFormer runner)
Results are saved under experiments/<exp_name>
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def create_experiment_dir(base_dir: str, exp_name: str) -> Path:
    exp_dir = Path(base_dir) / f"{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir: Path, config: dict):
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved: {config_file}")


def run_training(script: str, exp_dir: Path, args: dict):
    cmd = [sys.executable, script]
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
            universal_newlines=True,
        )
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        process.wait()
    return process.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run ESANet MTL experiment")
    parser.add_argument("--base-dir", type=str, default="./experiments")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="../dataset")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--loss-type", type=str, default="silog")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--encoder-rgb", type=str, default="resnet34")
    parser.add_argument("--encoder-depth", type=str, default="resnet34")
    parser.add_argument("--encoder-block", type=str, default="NonBottleneck1D")
    parser.add_argument("--pretrained-path", type=str, default="/home/shinds/my_document/DLFromScratch5/test/vae/sss/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth")
    parser.add_argument("--use-uncertainty-weighting", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    exp_name = args.exp_name or f"esanet_mtl_{args.encoder_rgb}"
    exp_dir = create_experiment_dir(args.base_dir, exp_name)

    print("=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"📁 Directory: {exp_dir}")
    print(f"🔬 Type: esanet_mtl")
    print("=" * 80)

    train_args = {
        "dataset-root": args.dataset_root,
        "output-dir": str(exp_dir),
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "num-workers": 4,
        "precision": args.precision,
        "accelerator": "gpu",
        "devices": "1",
        "gradient-clip-val": 1.0,
        "vis-max": 4,
        "loss-type": args.loss_type,
        "height": args.height,
        "width": args.width,
        "encoder-rgb": args.encoder_rgb,
        "encoder-depth": args.encoder_depth,
        "encoder-block": args.encoder_block,
        "pretrained-path": args.pretrained_path,
    }

    # Resolve training script path (this file's directory + train_esanet_mtl_real.py)
    train_script = str((Path(__file__).parent / "train_esanet_mtl_real.py").resolve())

    config = {
        "experiment_name": exp_name,
        "experiment_type": "esanet_mtl",
        "timestamp": datetime.now().isoformat(),
        "script": train_script,
        "arguments": train_args,
    }
    save_config(exp_dir, config)

    returncode = run_training(train_script, exp_dir, train_args)
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


