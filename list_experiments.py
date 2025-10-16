#!/usr/bin/env python3
"""
List and manage experiments
"""
import json
from pathlib import Path
from datetime import datetime
import argparse


def list_experiments(base_dir: str = "./experiments"):
    """List all experiments with their configurations"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"No experiments found in {base_dir}")
        return []
    
    experiments = []
    
    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        config_file = exp_dir / "config.json"
        if not config_file.exists():
            continue
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for results
        has_ckpt = len(list(exp_dir.glob("*.ckpt"))) > 0
        has_curves = (exp_dir / "curves").exists()
        has_vis = (exp_dir / "vis").exists()
        
        experiments.append({
            "dir": exp_dir.name,
            "path": str(exp_dir),
            "config": config,
            "completed": has_ckpt and has_curves,
        })
    
    return experiments


def print_experiments(experiments: list):
    """Print experiments in a table"""
    print("\n" + "=" * 100)
    print("EXPERIMENT HISTORY")
    print("=" * 100)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n{'#':<4} {'Name':<40} {'Type':<8} {'Encoder':<10} {'Status':<12}")
    print("-" * 100)
    
    for i, exp in enumerate(experiments, 1):
        config = exp["config"]
        name = config.get("experiment_name", "unknown")
        exp_type = config.get("experiment_type", "?")
        encoder = config["arguments"].get("encoder", "?")
        status = "✅ Complete" if exp["completed"] else "⏳ Running"
        
        print(f"{i:<4} {name:<40} {exp_type:<8} {encoder:<10} {status:<12}")
    
    print("-" * 100)
    print(f"Total: {len(experiments)} experiments")
    print("=" * 100)


def show_experiment_detail(exp_dir: str):
    """Show detailed information about an experiment"""
    exp_path = Path(exp_dir)
    config_file = exp_path / "config.json"
    
    if not config_file.exists():
        print(f"❌ No config found in {exp_dir}")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {config['experiment_name']}")
    print("=" * 80)
    
    print(f"\n📁 Directory: {exp_path}")
    print(f"🕐 Timestamp: {config['timestamp']}")
    print(f"🔬 Type: {config['experiment_type']}")
    print(f"📜 Script: {config['script']}")
    
    print(f"\n⚙️  Configuration:")
    for key, value in config['arguments'].items():
        print(f"   {key}: {value}")
    
    # Check for results
    print(f"\n📊 Results:")
    ckpts = list(exp_path.glob("*.ckpt"))
    print(f"   Checkpoints: {len(ckpts)}")
    
    if (exp_path / "curves").exists():
        curves = list((exp_path / "curves").glob("*.png"))
        print(f"   Curves: {len(curves)}")
    
    if (exp_path / "vis").exists():
        vis_files = list((exp_path / "vis").rglob("*.png"))
        print(f"   Visualizations: {len(vis_files)}")
    
    log_file = exp_path / "training.log"
    if log_file.exists():
        size_mb = log_file.stat().st_size / (1024 * 1024)
        print(f"   Log: {size_mb:.1f} MB")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="List and manage experiments")
    parser.add_argument("--base-dir", type=str, default="./experiments",
                       help="Base experiments directory")
    parser.add_argument("--detail", type=str, default=None,
                       help="Show detail for specific experiment directory")
    
    args = parser.parse_args()
    
    if args.detail:
        show_experiment_detail(args.detail)
    else:
        experiments = list_experiments(args.base_dir)
        print_experiments(experiments)
        
        if experiments:
            print("\nTo see details:")
            print(f"  python list_experiments.py --detail {experiments[-1]['path']}")


if __name__ == "__main__":
    main()

