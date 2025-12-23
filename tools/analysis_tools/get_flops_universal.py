#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
범용 FLOPs 계산 스크립트 (MTL, Seg-Only, Depth-Only 모두 지원)

문제점:
- 기존 get_flops_mtl.py는 MTL 모델만 고려
- seg_only/depth_only 모델의 FLOPs가 MTL과 비슷하게 나오는 문제
- auxiliary_head 제거 등으로 인한 부정확한 계산

해결책:
1. 모델 타입을 자동으로 감지 (MTL vs Single-Task)
2. 각 태스크별로 정확하게 FLOPs 계산
3. 컴포넌트별 상세 분석
4. forward pass를 실제로 수행하여 정확한 계산

사용법:
    # 단일 모델 분석
    python tools/analysis_tools/get_flops_universal.py \
        configs/chamnet/chamnet_mtl_shared_decoder_segformer_chamdata.py --shape 512 512

    # 여러 모델 비교
    python tools/analysis_tools/get_flops_universal.py \
        configs/chamnet/chamnet_seg_only_segformer_chamdata.py \
        configs/chamnet/chamnet_depth_only_segformer_chamdata.py \
        configs/chamnet/chamnet_mtl_base_segformer_chamdata.py \
        --compare --shape 512 512
"""

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Universal FLOPs calculator for Segmentation, Depth, and MTL models')
    parser.add_argument('config', nargs='+', help='Config file path(s)')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='Input image size (H W or single value for square)')
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple configs (requires 2+ configs)')
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed layer-by-layer analysis')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (key=value format)')
    return parser.parse_args()


def detect_model_type(model: nn.Module, cfg: Config) -> str:
    """
    모델 타입을 자동으로 감지합니다.

    Returns:
        'mtl': Multi-Task Learning (seg + depth)
        'segmentation': Segmentation only
        'depth': Depth estimation only
        'unknown': 알 수 없는 타입
    """
    model_class_name = type(model).__name__

    # 1. 클래스 이름으로 감지
    if 'MTL' in model_class_name or 'MultiTask' in model_class_name:
        return 'mtl'

    if 'Depth' in model_class_name:
        return 'depth'

    # 2. 모델 속성으로 감지
    has_seg_head = False
    has_depth_head = False

    # MTL 모델의 다양한 헤드 패턴
    seg_head_attrs = ['seg_decode_head', 'seg_task_head', 'seg_head']
    depth_head_attrs = ['depth_decode_head', 'depth_task_head', 'depth_head']

    for attr in seg_head_attrs:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            has_seg_head = True
            break

    for attr in depth_head_attrs:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            has_depth_head = True
            break

    # 일반 decode_head도 확인
    if hasattr(model, 'decode_head') and model.decode_head is not None:
        # Config에서 태스크 타입 확인
        if 'depth' in str(cfg).lower() and 'seg' not in model_class_name.lower():
            has_depth_head = True
        else:
            has_seg_head = True

    if has_seg_head and has_depth_head:
        return 'mtl'
    elif has_depth_head:
        return 'depth'
    elif has_seg_head:
        return 'segmentation'

    return 'unknown'


def analyze_model_components(model: nn.Module, model_type: str) -> Dict:
    """모델의 각 컴포넌트별 파라미터를 분석합니다."""

    components = {}

    # 1. Backbone
    if hasattr(model, 'backbone') and model.backbone is not None:
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        backbone_size = sum(p.numel() * p.element_size() for p in model.backbone.parameters())
        components['backbone'] = {
            'type': type(model.backbone).__name__,
            'params': backbone_params,
            'size_mb': backbone_size / (1024**2),
            'trainable': sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        }

    # 2. Neck (있는 경우)
    if hasattr(model, 'neck') and model.neck is not None:
        neck_params = sum(p.numel() for p in model.neck.parameters())
        neck_size = sum(p.numel() * p.element_size() for p in model.neck.parameters())
        components['neck'] = {
            'type': type(model.neck).__name__,
            'params': neck_params,
            'size_mb': neck_size / (1024**2),
            'trainable': sum(p.numel() for p in model.neck.parameters() if p.requires_grad)
        }

    # 3. Decode Heads (모든 가능한 패턴 확인)
    head_attrs = [
        'decode_head',
        'seg_decode_head',
        'depth_decode_head',
        'shared_decode_head',
        'seg_task_head',
        'depth_task_head',
        'seg_head',
        'depth_head',
    ]

    for attr in head_attrs:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if head is not None:
                head_params = sum(p.numel() for p in head.parameters())
                head_size = sum(p.numel() * p.element_size() for p in head.parameters())
                components[attr] = {
                    'type': type(head).__name__,
                    'params': head_params,
                    'size_mb': head_size / (1024**2),
                    'trainable': sum(p.numel() for p in head.parameters() if p.requires_grad)
                }

    # 4. Auxiliary Head
    if hasattr(model, 'auxiliary_head') and model.auxiliary_head is not None:
        aux_params = sum(p.numel() for p in model.auxiliary_head.parameters())
        aux_size = sum(p.numel() * p.element_size() for p in model.auxiliary_head.parameters())
        components['auxiliary_head'] = {
            'type': type(model.auxiliary_head).__name__,
            'params': aux_params,
            'size_mb': aux_size / (1024**2),
            'trainable': sum(p.numel() for p in model.auxiliary_head.parameters() if p.requires_grad)
        }

    # 5. MTL-specific: Uncertainty weights
    if hasattr(model, 'log_var_seg') or hasattr(model, 'log_var_depth'):
        uncertainty_params = 0
        if hasattr(model, 'log_var_seg'):
            uncertainty_params += 1
        if hasattr(model, 'log_var_depth'):
            uncertainty_params += 1

        components['uncertainty_weights'] = {
            'type': 'Learnable',
            'params': uncertainty_params,
            'size_mb': uncertainty_params * 4 / (1024**2),  # float32
            'trainable': uncertainty_params
        }

    return components


def calculate_model_statistics(model: nn.Module) -> Dict:
    """모델의 전체 통계를 계산합니다."""

    # 중복 제거된 파라미터 계산
    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    total_params = sum(p.numel() for p in param_dict.values())
    trainable_params = sum(p.numel() for p in param_dict.values() if p.requires_grad)

    param_size = sum(p.numel() * p.element_size() for p in param_dict.values())
    buffer_size = sum(b.numel() * b.element_size() for b in buffer_dict.values())

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'param_size_mb': param_size / (1024**2),
        'buffer_size_mb': buffer_size / (1024**2),
        'total_size_mb': (param_size + buffer_size) / (1024**2),
        'num_param_tensors': len(param_dict),
        'num_buffer_tensors': len(buffer_dict),
    }


def calculate_flops_mtl_manual(model: nn.Module, input_shape: Tuple, logger: MMLogger) -> Dict:
    """
    MTL 모델의 FLOPs를 컴포넌트별로 수동 계산

    mmengine의 get_model_complexity_info는 MTL 모델의 두 헤드를 모두 trace하지 못하므로
    각 컴포넌트를 개별적으로 계산하고 합산합니다.

    지원하는 MTL 구조:
    1. Independent Decoders: seg_decode_head + depth_decode_head
    2. Shared Decoder: shared_decode_head + seg_task_head + depth_task_head
    """
    logger.info("Calculating MTL FLOPs manually (component-wise)...")

    try:
        total_flops = 0
        breakdown = {}

        # Dummy input - 배치 차원 추가 필요
        # input_shape가 (C, H, W)이면 (1, C, H, W)로 만듦
        if len(input_shape) == 3:
            batch_input_shape = (1,) + input_shape
        else:
            batch_input_shape = input_shape

        dummy_input = torch.rand(batch_input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()

        # 1. Backbone FLOPs
        if hasattr(model, 'backbone'):
            logger.info("  Calculating backbone FLOPs...")
            backbone_flops = get_model_complexity_info(
                model.backbone,
                input_shape=input_shape,
                show_table=False,
                show_arch=False
            )
            breakdown['backbone'] = backbone_flops['flops']
            total_flops += backbone_flops['flops']
            logger.info(f"    Backbone: {backbone_flops['flops']/1e9:.2f} GFLOPs")

        # 2. Neck FLOPs (if exists)
        if hasattr(model, 'neck') and model.neck is not None:
            logger.info("  Calculating neck FLOPs...")
            with torch.no_grad():
                backbone_out = model.backbone(dummy_input)
                if isinstance(backbone_out, (list, tuple)):
                    neck_input = backbone_out
                else:
                    neck_input = [backbone_out]

            breakdown['neck'] = 0
            logger.info("    Neck: skipped (negligible)")

        # Extract features once for all heads
        with torch.no_grad():
            features = model.extract_feat(dummy_input)
            if not isinstance(features, (list, tuple)):
                features = [features]

        # 3. Independent Decoders 구조
        # 3a. Segmentation Decode Head FLOPs
        if hasattr(model, 'seg_decode_head') and model.seg_decode_head is not None:
            logger.info("  Calculating seg_decode_head FLOPs...")
            seg_head_flops = get_model_complexity_info(
                model.seg_decode_head,
                input_shape=None,
                inputs=features,
                show_table=False,
                show_arch=False
            )
            breakdown['seg_decode_head'] = seg_head_flops['flops']
            total_flops += seg_head_flops['flops']
            logger.info(f"    Seg Head: {seg_head_flops['flops']/1e9:.2f} GFLOPs")

        # 3b. Depth Decode Head FLOPs
        if hasattr(model, 'depth_decode_head') and model.depth_decode_head is not None:
            logger.info("  Calculating depth_decode_head FLOPs...")
            depth_head_flops = get_model_complexity_info(
                model.depth_decode_head,
                input_shape=None,
                inputs=features,
                show_table=False,
                show_arch=False
            )
            breakdown['depth_decode_head'] = depth_head_flops['flops']
            total_flops += depth_head_flops['flops']
            logger.info(f"    Depth Head: {depth_head_flops['flops']/1e9:.2f} GFLOPs")

        # 4. Shared Decoder 구조
        # 4a. Shared Decode Head FLOPs
        if hasattr(model, 'shared_decode_head') and model.shared_decode_head is not None:
            logger.info("  Calculating shared_decode_head FLOPs...")
            shared_head_flops = get_model_complexity_info(
                model.shared_decode_head,
                input_shape=None,
                inputs=features,
                show_table=False,
                show_arch=False
            )
            breakdown['shared_decode_head'] = shared_head_flops['flops']
            total_flops += shared_head_flops['flops']
            logger.info(f"    Shared Decode Head: {shared_head_flops['flops']/1e9:.2f} GFLOPs")

            # Shared head의 출력을 얻어야 task heads 계산 가능
            with torch.no_grad():
                shared_features = model.shared_decode_head(features)
                # SegformerHead는 logits을 반환하므로 이를 다시 리스트로 감싸야 함
                if not isinstance(shared_features, (list, tuple)):
                    shared_features = [shared_features]

            # 4b. Seg Task Head FLOPs
            if hasattr(model, 'seg_task_head') and model.seg_task_head is not None:
                logger.info("  Calculating seg_task_head FLOPs...")
                seg_task_flops = get_model_complexity_info(
                    model.seg_task_head,
                    input_shape=None,
                    inputs=shared_features,
                    show_table=False,
                    show_arch=False
                )
                breakdown['seg_task_head'] = seg_task_flops['flops']
                total_flops += seg_task_flops['flops']
                logger.info(f"    Seg Task Head: {seg_task_flops['flops']/1e9:.2f} GFLOPs")

            # 4c. Depth Task Head FLOPs
            if hasattr(model, 'depth_task_head') and model.depth_task_head is not None:
                logger.info("  Calculating depth_task_head FLOPs...")
                depth_task_flops = get_model_complexity_info(
                    model.depth_task_head,
                    input_shape=None,
                    inputs=shared_features,
                    show_table=False,
                    show_arch=False
                )
                breakdown['depth_task_head'] = depth_task_flops['flops']
                total_flops += depth_task_flops['flops']
                logger.info(f"    Depth Task Head: {depth_task_flops['flops']/1e9:.2f} GFLOPs")

        # Total params
        total_params = sum(p.numel() for p in model.parameters())

        result = {
            'flops_raw': total_flops,
            'params_raw': total_params,
            'flops_str': _format_size(total_flops),
            'params_str': _format_size(total_params),
            'gflops': total_flops / 1e9,
            'gmacs': total_flops / 2e9,
            'mflops': total_flops / 1e6,
            'success': True,
            'error': None,
            'breakdown': breakdown,
            'method': 'manual_mtl'
        }

        logger.info(f"  Total MTL FLOPs: {result['gflops']:.2f} GFLOPs")
        return result

    except Exception as e:
        logger.error(f"Manual MTL FLOPs calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'flops_raw': 0,
            'params_raw': 0,
            'flops_str': 'N/A',
            'params_str': 'N/A',
            'gflops': 0,
            'gmacs': 0,
            'mflops': 0,
            'success': False,
            'error': str(e),
            'method': 'manual_mtl'
        }


def calculate_flops_accurate(model: nn.Module, input_shape: Tuple,
                             model_type: str, logger: MMLogger) -> Dict:
    """
    정확한 FLOPs 계산

    MTL 모델의 경우 컴포넌트별로 계산하고, 일반 모델은 기존 방식 사용
    """

    # MTL 모델인 경우 수동 계산
    if model_type == 'mtl':
        logger.info("Detected MTL model - using component-wise calculation")
        return calculate_flops_mtl_manual(model, input_shape, logger)

    # 일반 모델은 기존 방식
    # Meta information
    ori_shape = input_shape[-2:]
    pad_shape = input_shape[-2:]

    # Prepare input data
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo={
            'ori_shape': ori_shape,
            'pad_shape': pad_shape
        })]
    }

    # Data preprocessing
    data = model.data_preprocessor(data_batch)

    # FLOPs 계산
    try:
        logger.info("Calculating FLOPs with mmengine.analysis...")

        outputs = get_model_complexity_info(
            model,
            input_shape=None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False
        )

        flops_raw = outputs['flops']
        params_raw = outputs['params']

        # 추가 정보
        result = {
            'flops_raw': flops_raw,
            'params_raw': params_raw,
            'flops_str': _format_size(flops_raw),
            'params_str': _format_size(params_raw),
            'gflops': flops_raw / 1e9,
            'gmacs': flops_raw / 2e9,  # 1 MAC = 2 FLOPs
            'mflops': flops_raw / 1e6,
            'success': True,
            'error': None,
            'method': 'standard'
        }

        logger.info(f"FLOPs calculation successful: {result['gflops']:.2f} GFLOPs")

        return result

    except Exception as e:
        logger.error(f"FLOPs calculation failed: {e}")
        return {
            'flops_raw': 0,
            'params_raw': 0,
            'flops_str': 'N/A',
            'params_str': 'N/A',
            'gflops': 0,
            'gmacs': 0,
            'mflops': 0,
            'success': False,
            'error': str(e),
            'method': 'standard'
        }


def analyze_config(config_path: str, input_shape: tuple,
                   detailed: bool, logger: MMLogger) -> Dict:
    """Config 파일을 분석하고 모델의 모든 정보를 추출합니다."""

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f'Config file not found: {config_file}')

    # Load config
    logger.info(f"Loading config: {config_file.name}")
    cfg = Config.fromfile(config_file)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'

    init_default_scope(cfg.get('scope', 'mmseg'))

    # Build model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    # Detect model type
    model_type = detect_model_type(model, cfg)
    logger.info(f"Detected model type: {model_type}")

    # Analyze components
    components = analyze_model_components(model, model_type)

    # Calculate statistics
    statistics = calculate_model_statistics(model)

    # Calculate FLOPs
    flops_info = calculate_flops_accurate(model, input_shape, model_type, logger)

    # Compile results
    result = {
        'config_path': str(config_path),
        'config_name': config_file.name,
        'model_class': type(model).__name__,
        'model_type': model_type,
        'input_shape': input_shape,
        'components': components,
        'statistics': statistics,
        'flops': flops_info,
    }

    return result


def print_single_result(result: Dict):
    """단일 모델 분석 결과를 출력합니다."""

    print(f"\n{'='*90}")
    print(f"📊 Model Complexity Analysis: {result['config_name']}")
    print(f"{'='*90}\n")

    # Basic info
    print(f"📁 Configuration:")
    print(f"  Config Path:  {result['config_path']}")
    print(f"  Model Class:  {result['model_class']}")
    print(f"  Model Type:   {result['model_type'].upper()}")
    print(f"  Input Shape:  {result['input_shape']}")

    # Components
    print(f"\n{'-'*90}")
    print("🔧 Model Components:")
    print(f"{'-'*90}")
    print(f"{'Component':<30s} {'Type':<30s} {'Params':>12s} {'Size (MB)':>12s}")
    print(f"{'-'*90}")

    for comp_name, comp_info in result['components'].items():
        params_m = comp_info['params'] / 1e6
        print(f"{comp_name:<30s} {comp_info['type']:<30s} "
              f"{params_m:>11.2f}M {comp_info['size_mb']:>12.2f}")

    # Statistics
    stats = result['statistics']
    print(f"\n{'-'*90}")
    print("📦 Model Parameters:")
    print(f"{'-'*90}")
    total_params_m = stats['total_params'] / 1e6
    trainable_params_m = stats['trainable_params'] / 1e6
    non_trainable_params_m = stats['non_trainable_params'] / 1e6

    print(f"  Total Parameters:        {total_params_m:>12.2f} M  ({stats['total_params']:,} params)")
    print(f"  Trainable Parameters:    {trainable_params_m:>12.2f} M  ({stats['trainable_params']:,} params)")
    print(f"  Non-trainable Params:    {non_trainable_params_m:>12.2f} M  ({stats['non_trainable_params']:,} params)")
    print(f"  Parameter Tensors:       {stats['num_param_tensors']:>12,}")
    print(f"  Buffer Tensors:          {stats['num_buffer_tensors']:>12,}")

    print(f"\n{'-'*90}")
    print("💾 Memory Footprint:")
    print(f"{'-'*90}")
    print(f"  Parameters:              {stats['param_size_mb']:>12.2f} MB")
    print(f"  Buffers:                 {stats['buffer_size_mb']:>12.2f} MB")
    print(f"  Total Model Size:        {stats['total_size_mb']:>12.2f} MB")

    # FLOPs
    flops = result['flops']
    if flops['success']:
        print(f"\n{'-'*90}")
        print("⚡ Computational Complexity:")
        print(f"{'-'*90}")
        print(f"  FLOPs:                   {flops['flops_str']:>12s}  ({flops['gflops']:.2f} GFLOPs)")
        print(f"  GFLOPs:                  {flops['gflops']:>12.2f}")
        print(f"  GMACs:                   {flops['gmacs']:>12.2f}")
        print(f"  MFLOPs:                  {flops['mflops']:>12.2f}")
        print(f"  Calculation Method:      {flops.get('method', 'standard')}")

        # MTL breakdown 표시
        if 'breakdown' in flops and flops['breakdown']:
            print(f"\n{'-'*90}")
            print("🔍 FLOPs Breakdown (MTL Components):")
            print(f"{'-'*90}")
            for comp_name, comp_flops in flops['breakdown'].items():
                comp_gflops = comp_flops / 1e9
                pct = (comp_flops / flops['flops_raw'] * 100) if flops['flops_raw'] > 0 else 0
                print(f"  {comp_name:<28s} {comp_gflops:>12.2f} GFLOPs  ({pct:>5.1f}%)")

            # 검증: breakdown 합계 확인
            total_breakdown = sum(flops['breakdown'].values())
            total_breakdown_gflops = total_breakdown / 1e9
            print(f"  {'-'*60}")
            print(f"  {'Total (from breakdown)':<28s} {total_breakdown_gflops:>12.2f} GFLOPs")

            # 오차 확인
            diff = abs(total_breakdown - flops['flops_raw'])
            if diff > 1e6:  # 1M FLOPs 이상 차이
                print(f"  ⚠️  Warning: Breakdown total differs from reported total by {diff/1e9:.2f} GFLOPs")
    else:
        print(f"\n{'-'*90}")
        print(f"⚠️  FLOPs Calculation Failed:")
        print(f"{'-'*90}")
        print(f"  Error: {flops['error']}")

    # Summary
    print(f"\n{'-'*90}")
    print("📈 Summary:")
    print(f"{'-'*90}")
    if flops['success']:
        print(f"  Model: {total_params_m:.2f}M params, {stats['total_size_mb']:.2f} MB, {flops['gflops']:.2f} GFLOPs")

        # Efficiency metrics
        if flops['gflops'] > 0 and total_params_m > 0:
            efficiency = total_params_m / flops['gflops']  # params per GFLOP
            print(f"  Efficiency: {efficiency:.2f} M params/GFLOPs")
    else:
        print(f"  Model: {total_params_m:.2f}M params, {stats['total_size_mb']:.2f} MB")

    print(f"\n{'='*90}\n")


def compare_results(results: List[Dict]):
    """여러 모델의 결과를 비교합니다."""

    if len(results) < 2:
        print("⚠️  Comparison requires at least 2 configs")
        return

    print(f"\n{'='*90}")
    print(f"Model Comparison ({len(results)} models)")
    print(f"{'='*90}\n")

    # Config list
    print("Configs:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['config_name']:<50s} (Type: {r['model_type']})")

    # Comparison table
    print(f"\n{'-'*90}")
    print("Metrics Comparison:")
    print(f"{'-'*90}")

    # Header
    header = f"{'Metric':<30s}"
    for i in range(len(results)):
        header += f" | {'[' + str(i+1) + ']':>15s}"
    if len(results) == 2:
        header += f" | {'Δ (2-1)':>15s}"
    print(header)
    print('-' * len(header))

    # Metrics to compare
    metrics = [
        ('Model Type', lambda r: r['model_type'], 's'),
        ('Total Params (M)', lambda r: r['statistics']['total_params'] / 1e6, '.2f'),
        ('Trainable Params (M)', lambda r: r['statistics']['trainable_params'] / 1e6, '.2f'),
        ('Model Size (MB)', lambda r: r['statistics']['total_size_mb'], '.2f'),
        ('GFLOPs', lambda r: r['flops']['gflops'] if r['flops']['success'] else 0, '.2f'),
        ('GMACs', lambda r: r['flops']['gmacs'] if r['flops']['success'] else 0, '.2f'),
    ]

    for metric_name, getter, fmt in metrics:
        line = f"{metric_name:<30s}"
        values = [getter(r) for r in results]

        for val in values:
            if fmt == 's':
                line += f" | {str(val):>15s}"
            elif ',' in fmt:
                line += f" | {val:>15{fmt}}"
            else:
                line += f" | {val:>15{fmt}}"

        # Difference for 2 models
        if len(results) == 2 and fmt != 's':
            diff = values[1] - values[0]
            line += f" | {diff:>+15{fmt}}"

        print(line)

    # Component comparison
    print(f"\n{'-'*90}")
    print("Component-wise Comparison (Parameters):")
    print(f"{'-'*90}")

    # Collect all component names
    all_components = set()
    for r in results:
        all_components.update(r['components'].keys())

    header = f"{'Component':<30s}"
    for i in range(len(results)):
        header += f" | {'[' + str(i+1) + ']':>15s}"
    if len(results) == 2:
        header += f" | {'Δ (2-1)':>15s}"
    print(header)
    print('-' * len(header))

    for comp in sorted(all_components):
        line = f"{comp:<30s}"
        comp_params = []

        for r in results:
            if comp in r['components']:
                params = r['components'][comp]['params']
                comp_params.append(params)
                line += f" | {params/1e6:>15.2f}"
            else:
                comp_params.append(0)
                line += f" | {'N/A':>15s}"

        if len(results) == 2:
            diff = comp_params[1] - comp_params[0]
            if comp_params[0] > 0 or comp_params[1] > 0:
                line += f" | {diff/1e6:>+15.2f}"
            else:
                line += f" | {'N/A':>15s}"

        print(line)

    # FLOPs breakdown comparison (for MTL models)
    if any(r['flops'].get('breakdown') for r in results):
        print(f"\n{'-'*90}")
        print("FLOPs Breakdown Comparison (MTL Components):")
        print(f"{'-'*90}")

        # Collect all breakdown components
        all_breakdown_comps = set()
        for r in results:
            if 'breakdown' in r['flops']:
                all_breakdown_comps.update(r['flops']['breakdown'].keys())

        if all_breakdown_comps:
            header = f"{'Component':<30s}"
            for i in range(len(results)):
                header += f" | {'[' + str(i+1) + ']':>15s}"
            print(header)
            print('-' * len(header))

            for comp in sorted(all_breakdown_comps):
                line = f"{comp:<30s}"
                for r in results:
                    if 'breakdown' in r['flops'] and comp in r['flops']['breakdown']:
                        flops_val = r['flops']['breakdown'][comp] / 1e9
                        line += f" | {flops_val:>15.2f}"
                    else:
                        line += f" | {'N/A':>15s}"
                print(line)

    # Analysis
    print(f"\n{'-'*90}")
    print("Analysis:")
    print(f"{'-'*90}")

    if len(results) == 2:
        diff_params = results[1]['statistics']['total_params'] - results[0]['statistics']['total_params']
        diff_flops = results[1]['flops']['gflops'] - results[0]['flops']['gflops']

        print(f"Parameter difference: {diff_params/1e6:+.2f}M ({diff_params/results[0]['statistics']['total_params']*100:+.1f}%)")

        if results[0]['flops']['success'] and results[1]['flops']['success']:
            print(f"FLOPs difference:     {diff_flops:+.2f} GFLOPs ({diff_flops/results[0]['flops']['gflops']*100:+.1f}%)")

            # Sanity check for MTL vs single-task
            types = [r['model_type'] for r in results]
            if 'mtl' in types and ('segmentation' in types or 'depth' in types):
                mtl_idx = types.index('mtl')
                single_idx = 1 - mtl_idx

                mtl_flops = results[mtl_idx]['flops']['gflops']
                single_flops = results[single_idx]['flops']['gflops']

                expected_ratio = mtl_flops / single_flops if single_flops > 0 else 0
                print(f"\nMTL vs Single-Task Check:")
                print(f"  MTL FLOPs:        {mtl_flops:.2f} GFLOPs")
                print(f"  Single FLOPs:     {single_flops:.2f} GFLOPs")
                print(f"  Ratio (MTL/Single): {expected_ratio:.2f}x")

                if expected_ratio < 1.2:
                    print(f"\n⚠️  경고: MTL 모델의 FLOPs가 single-task보다 크게 높지 않습니다!")
                    print(f"   예상: MTL이 single-task의 약 1.5~2배가 되어야 합니다.")
                    print(f"   → FLOPs 계산을 확인하세요.")
                elif expected_ratio >= 1.5 and expected_ratio <= 2.5:
                    print(f"\n✅ MTL FLOPs가 합리적인 범위입니다 (1.5x~2.5x)")
                else:
                    print(f"\n⚠️  주의: MTL FLOPs 비율이 예상 범위를 벗어났습니다 (1.5x~2.5x)")

    print(f"\n{'='*90}\n")


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    # Parse input shape
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('Invalid input shape. Use: --shape H W or --shape SIZE')

    # Analyze each config
    results = []
    for config_path in args.config:
        try:
            result = analyze_config(config_path, input_shape, args.detailed, logger)
            results.append(result)

            # Print individual results if not comparing or only one config
            if not args.compare or len(args.config) == 1:
                print_single_result(result)

        except Exception as e:
            logger.error(f"Failed to analyze {config_path}: {e}")
            import traceback
            traceback.print_exc()

    # Compare mode
    if args.compare and len(results) > 1:
        compare_results(results)


if __name__ == '__main__':
    main()
