#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
Multi-Task Learning (MTL) Full Pipeline 자동 실행 스크립트

학습 → 테스트 → 벤치마크 → FLOPs 계산 (Universal) → Learning Curve Plot을 순차적으로 실행합니다.

📊 FLOPs 계산: get_flops_universal.py 사용 (MTL/Seg/Depth 모두 지원)



사용법:
    #  Uncertainty Weighting 
    python tools/run_full_pipeline_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdata.py --weight-strategy uncertainty

    # DWA 
    python tools/run_full_pipeline_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdata.py --weight-strategy dwa

    # Manual Weighting 
    python tools/run_full_pipeline_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdatav4.py \
        --weight-strategy manual --seg-weight 1.0 --depth-weight 5.0 
"""

import argparse
import glob
import os
import os.path as osp
import subprocess
import sys

from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='MTL Full Pipeline: train -> test -> benchmark -> flops -> plot')

    # 기본 arguments
    parser.add_argument('config', help='MTL training config file path')
    parser.add_argument('--work-dir', help='work directory (default: auto-determined)')
    parser.add_argument('--skip-train', action='store_true', help='skip training step')
    parser.add_argument('--skip-test', action='store_true', help='skip testing step')
    parser.add_argument('--skip-benchmark', action='store_true', help='skip benchmark step')
    parser.add_argument('--skip-flops', action='store_true', help='skip FLOPs calculation step')
    parser.add_argument('--benchmark-repeat-times', type=int, default=10,
                        help='number of times to repeat benchmark (default: 10)')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--amp', action='store_true', help='enable automatic-mixed-precision training')

    # ========== MTL-specific arguments ==========
    # Loss weighting strategy
    parser.add_argument(
        '--weight-strategy',
        type=str,
        choices=['dwa', 'uncertainty', 'manual', 'fixed'],
        default='uncertainty',
        help='Loss weighting strategy')

    # DWA arguments
    parser.add_argument('--dwa-temp', type=float, help='DWA temperature (default: 2.0)')
    parser.add_argument('--dwa-window', type=int, help='DWA window size (default: 10)')
    parser.add_argument('--dwa-update-freq', type=int, help='DWA update frequency (default: 50)')

    # Uncertainty Weighting arguments
    parser.add_argument('--uncertainty-init-log-var-seg', type=float,
                        help='Initial log variance for segmentation (default: 0.0)')
    parser.add_argument('--uncertainty-init-log-var-depth', type=float,
                        help='Initial log variance for depth (default: 0.0)')

    # Manual/Fixed weighting arguments
    parser.add_argument('--seg-weight', type=float, help='Segmentation loss weight (default: 1.0)')
    parser.add_argument('--depth-weight', type=float, help='Depth loss weight (default: 1.0)')

    args = parser.parse_args()
    return args


def find_best_mtl_checkpoint(work_dir):
    """work_dir에서 best MTL checkpoint 파일 찾기
    
    MTL은 seg/mIoU와 depth/abs_rel 두 개의 best checkpoint를 저장함.
    기본적으로 depth/abs_rel 기준 checkpoint를 반환 (더 학습하기 어려운 task)
    """
    # 우선순위: depth > seg > any best
    patterns = [
        osp.join(work_dir, 'best_depth_abs_rel_iter_*.pth'),  # Depth 기준 (primary)
        osp.join(work_dir, 'best_seg_mIoU_iter_*.pth'),       # Seg 기준
        osp.join(work_dir, 'best_mIoU_iter_*.pth'),           # Fallback
        osp.join(work_dir, '*best*.pth'),                      # Any best
    ]

    for pattern in patterns:
        checkpoints = glob.glob(pattern)
        if checkpoints:
            checkpoint = max(checkpoints, key=osp.getmtime)
            return checkpoint

    raise FileNotFoundError(
        f'Best MTL checkpoint not found in {work_dir}. '
        f'Expected: best_depth_abs_rel_iter_*.pth or best_seg_mIoU_iter_*.pth')


def get_work_dir_from_config_and_args(config_path, args):
    """Config 파일과 arguments로 work_dir 결정
    
    우선순위:
    1. CLI --work-dir argument
    2. Config 파일의 work_dir 설정
    3. 자동 생성 (config_basename + weight_strategy)
    """
    # 1. CLI argument가 최우선
    if args.work_dir:
        return args.work_dir

    # 2. Config 파일에서 work_dir 읽기
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, 'work_dir') and cfg.work_dir:
        return cfg.work_dir
    
    # 3. Fallback: 자동 생성
    config_basename = osp.splitext(osp.basename(config_path))[0]
    strategy_suffix = args.weight_strategy
    return osp.join('./work_dirs', f'{config_basename}_{strategy_suffix}')


def run_command(cmd, description, log_path=None):
    """명령어 실행 및 에러 처리"""
    print("=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    if log_path:
        print(f"📁 Log: {log_path}")
    print("=" * 80)

    if log_path:
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            result = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"✅ {description} completed successfully")
    print("=" * 80)
    print()


def build_mtl_train_args(args):
    """MTL training arguments 생성"""
    cmd = [sys.executable, 'tools/train_mtl.py', args.config]

    # Weight strategy
    cmd.extend(['--weight-strategy', args.weight_strategy])

    # DWA arguments
    if args.weight_strategy == 'dwa':
        if args.dwa_temp is not None:
            cmd.extend(['--dwa-temp', str(args.dwa_temp)])
        if args.dwa_window is not None:
            cmd.extend(['--dwa-window', str(args.dwa_window)])
        if args.dwa_update_freq is not None:
            cmd.extend(['--dwa-update-freq', str(args.dwa_update_freq)])

    # Uncertainty arguments
    elif args.weight_strategy == 'uncertainty':
        if args.uncertainty_init_log_var_seg is not None:
            cmd.extend(['--uncertainty-init-log-var-seg', str(args.uncertainty_init_log_var_seg)])
        if args.uncertainty_init_log_var_depth is not None:
            cmd.extend(['--uncertainty-init-log-var-depth', str(args.uncertainty_init_log_var_depth)])

    # Manual/Fixed arguments
    elif args.weight_strategy in ['manual', 'fixed']:
        if args.seg_weight is not None:
            cmd.extend(['--seg-weight', str(args.seg_weight)])
        if args.depth_weight is not None:
            cmd.extend(['--depth-weight', str(args.depth_weight)])

    # Work dir
    if args.work_dir:
        cmd.extend(['--work-dir', args.work_dir])

    # Resume and AMP
    if args.resume:
        cmd.append('--resume')
    if args.amp:
        cmd.append('--amp')

    return cmd


def main():
    args = parse_args()

    config_path = osp.abspath(args.config)
    if not osp.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    # work_dir 결정
    work_dir = get_work_dir_from_config_and_args(config_path, args)
    work_dir = osp.abspath(work_dir)

    print("=" * 80)
    print("📋 Multi-Task Learning (MTL) Full Pipeline")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    print(f"Weight Strategy: {args.weight_strategy}")
    print("=" * 80)
    print()

    # 1. Training
    if not args.skip_train:
        train_cmd = build_mtl_train_args(args)
        if '--work-dir' not in train_cmd:
            train_cmd.extend(['--work-dir', work_dir])
        run_command(train_cmd, "MTL Training")
    else:
        print("⏭️  Skipping training step")
        print()

    # 2. Find best checkpoint
    print("🔍 Searching for best MTL checkpoint...")
    try:
        best_checkpoint = find_best_mtl_checkpoint(work_dir)
        print(f"✅ Found best checkpoint: {best_checkpoint}")
        print()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        if not args.skip_train:
            print("Training may have failed or checkpoint not saved yet.")
            sys.exit(1)
        else:
            print("Please provide checkpoint path manually or run training first.")
            sys.exit(1)

    # 3. Testing (시각화 완전 비활성화)
    # NOTE: 시각화는 mmseg 의존성 문제로 오류 발생 가능
    # 평가만 수행하고, 시각화는 나중에 별도로 추론할 때 진행
    if not args.skip_test:
        # Config 옵션을 한 줄씩 구분하여 전달 (더 명확한 override)
        test_cmd = [
            sys.executable, 'tools/test.py',
            config_path,
            best_checkpoint,
            '--cfg-options',
            # Custom hooks 완전 비활성화 (MTLVisualizationHook 등 제거)
            'custom_hooks=[]',
            # Default visualization hook도 비활성화
            'default_hooks.visualization.draw=False',
            'default_hooks.visualization.interval=999999999',
        ]
        run_command(test_cmd, "MTL Testing (Evaluation Only)")
    else:
        print("⏭️  Skipping test step")
        print()

    # 4. Benchmark
    if not args.skip_benchmark:
        benchmark_cmd = [
            sys.executable, 'tools/analysis_tools/benchmark.py',
            config_path,
            best_checkpoint,
            '--repeat-times', str(args.benchmark_repeat_times),
            '--work-dir', work_dir
        ]
        run_command(benchmark_cmd, "Benchmark (FPS calculation)")
    else:
        print("⏭️  Skipping benchmark step")
        print()

    # 5. FLOPs calculation (Universal - MTL/Seg/Depth 모두 지원)
    if not args.skip_flops:
        flops_dir = osp.join(work_dir, 'flops')

        flops_universal_cmd = [
            sys.executable, 'tools/analysis_tools/get_flops_universal.py',
            config_path,
            '--shape', '512', '512'
        ]
        flops_universal_log = osp.join(flops_dir, 'get_flops_universal.log')
        run_command(flops_universal_cmd, "FLOPs calculation (Universal)", log_path=flops_universal_log)
    else:
        print("⏭️  Skipping FLOPs calculation step")
        print()

    # 6. Plot Learning Curve (MTL 전용)
    learning_curve_csv = osp.join(work_dir, 'learning_curve.csv')
    if osp.exists(learning_curve_csv):
        plot_cmd = [
            sys.executable, 'tools/plot_mtl_learning_curve.py',
            work_dir,
            '--dpi', '150',
            '--smooth', '5'
        ]
        run_command(plot_cmd, "MTL Learning Curve Plotting")
    else:
        print("⚠️  learning_curve.csv not found, skipping plot")
    print()

    # 최종 요약
    print("=" * 80)
    print("✅ MTL Full Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    print(f"Weight Strategy: {args.weight_strategy}")
    print(f"Best Checkpoint: {best_checkpoint}")
    print()
    print("📊 Results:")
    print(f"  - Test Results: {work_dir}/test/")
    print(f"  - Learning Curve: {learning_curve_csv}")
    print(f"  - Benchmark: {work_dir}/benchmark/")
    print(f"  - FLOPs (Universal): {work_dir}/flops/get_flops_universal.log")
    print()
    print("💡 FLOPs 계산:")
    print("   - get_flops_universal.py 사용 (MTL, Segmentation, Depth 모두 지원)")
    print("   - 모델 타입 자동 감지 및 컴포넌트별 상세 분석")
    print("   - 정확한 forward pass 기반 계산")
    print("=" * 80)

if __name__ == '__main__':
    main()
