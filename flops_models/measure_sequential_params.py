#!/usr/bin/env python3
# ============================================================================
# ESANet Sequential Models Parameter Count Measurement Script
# ============================================================================
# 이 스크립트는 순차적으로 실행되는 ESANet 모델들의 파라미터 수를 측정합니다.
# 주요 기능:
# - 의미 분할 모델과 깊이 추정 모델의 개별 파라미터 수 측정
# - 순차 실행 시 총 파라미터 수 계산
# - 메모리 사용량 분석
# - 모델 복잡도 비교

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 실제 학습 코드에서 모델 import
# ============================================================================
try:
    from models.train_esanet_seg_only import LightningESANetSegOnly
    SEG_MODEL_AVAILABLE = True
    print("✅ LightningESANetSegOnly 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ LightningESANetSegOnly 모델 import 실패: {e}")
    SEG_MODEL_AVAILABLE = False

try:
    from models.train_esanet_depth_only import LightningESANetDepthOnly
    DEPTH_MODEL_AVAILABLE = True
    print("✅ LightningESANetDepthOnly 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ LightningESANetDepthOnly 모델 import 실패: {e}")
    DEPTH_MODEL_AVAILABLE = False

try:
    from models.train_esanet_mtl_uncertain import LightningESANetMTL
    MTL_MODEL_AVAILABLE = True
    print("✅ LightningESANetMTL 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ LightningESANetMTL 모델 import 실패: {e}")
    MTL_MODEL_AVAILABLE = False


# ============================================================================
# 파라미터 수 측정 함수들
# ============================================================================
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    모델의 파라미터 수를 계산합니다.
    
    Args:
        model: 측정할 모델
        
    Returns:
        Dict[str, int]: 파라미터 수 정보
            - total: 총 파라미터 수
            - trainable: 학습 가능한 파라미터 수
            - frozen: 고정된 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }


def format_number(num: int) -> str:
    """
    숫자를 읽기 쉬운 형태로 포맷팅합니다.
    
    Args:
        num: 포맷팅할 숫자
        
    Returns:
        str: 포맷팅된 문자열 (예: "15.2M", "1.5B")
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    모델의 메모리 사용량을 추정합니다.
    
    Args:
        model: 측정할 모델
        input_shape: 입력 텐서의 형태
        
    Returns:
        Dict[str, float]: 메모리 사용량 정보 (MB)
            - model_params: 모델 파라미터 메모리
            - model_buffers: 모델 버퍼 메모리
            - input_memory: 입력 텐서 메모리
            - total: 총 메모리 사용량
    """
    # 모델 파라미터 메모리 (float32 기준)
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # MB
    
    # 모델 버퍼 메모리 (BatchNorm 통계 등)
    buffer_memory = sum(b.numel() * 4 for b in model.buffers()) / (1024 * 1024)  # MB
    
    # 입력 텐서 메모리
    input_memory = torch.tensor(input_shape).numel() * 4 / (1024 * 1024)  # MB
    
    total_memory = param_memory + buffer_memory + input_memory
    
    return {
        "model_params": param_memory,
        "model_buffers": buffer_memory,
        "input_memory": input_memory,
        "total": total_memory
    }


def load_model_from_checkpoint(checkpoint_path: str, model_class, device: str = "cpu") -> nn.Module:
    """
    체크포인트에서 모델을 로드합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_class: 모델 클래스
        device: 디바이스
        
    Returns:
        nn.Module: 로드된 모델
    """
    try:
        model = model_class.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return None


def measure_sequential_models(seg_ckpt_path: str, depth_ckpt_path: str, 
                            height: int = 512, width: int = 512,
                            device: str = "cpu") -> None:
    """
    순차적으로 실행되는 모델들의 파라미터 수를 측정합니다.
    
    Args:
        seg_ckpt_path: 의미 분할 모델 체크포인트 경로
        depth_ckpt_path: 깊이 추정 모델 체크포인트 경로
        height: 입력 이미지 높이
        width: 입력 이미지 너비
        device: 디바이스
    """
    print("="*80)
    print("ESANet 순차 모델 파라미터 수 측정")
    print("="*80)
    
    # 1. 의미 분할 모델 측정
    print("\n🔍 의미 분할 모델 분석...")
    if SEG_MODEL_AVAILABLE:
        seg_model = load_model_from_checkpoint(seg_ckpt_path, LightningESANetSegOnly, device)
        if seg_model is not None:
            seg_params = count_parameters(seg_model)
            seg_memory = estimate_memory_usage(seg_model, (1, 3, height, width))
            
            print(f"  📊 파라미터 수:")
            print(f"    - 총 파라미터: {format_number(seg_params['total'])} ({seg_params['total']:,})")
            print(f"    - 학습 가능: {format_number(seg_params['trainable'])} ({seg_params['trainable']:,})")
            print(f"    - 고정됨: {format_number(seg_params['frozen'])} ({seg_params['frozen']:,})")
            print(f"  💾 메모리 사용량:")
            print(f"    - 모델 파라미터: {seg_memory['model_params']:.2f} MB")
            print(f"    - 모델 버퍼: {seg_memory['model_buffers']:.2f} MB")
            print(f"    - 입력 텐서: {seg_memory['input_memory']:.2f} MB")
            print(f"    - 총 메모리: {seg_memory['total']:.2f} MB")
        else:
            print("  ❌ 의미 분할 모델 로드 실패")
            seg_params = {"total": 0, "trainable": 0, "frozen": 0}
            seg_memory = {"total": 0}
    else:
        print("  ❌ 의미 분할 모델을 사용할 수 없습니다")
        seg_params = {"total": 0, "trainable": 0, "frozen": 0}
        seg_memory = {"total": 0}
    
    # 2. 깊이 추정 모델 측정
    print("\n🔍 깊이 추정 모델 분석...")
    if DEPTH_MODEL_AVAILABLE:
        depth_model = load_model_from_checkpoint(depth_ckpt_path, LightningESANetDepthOnly, device)
        if depth_model is not None:
            depth_params = count_parameters(depth_model)
            depth_memory = estimate_memory_usage(depth_model, (1, 3, height, width))
            
            print(f"  📊 파라미터 수:")
            print(f"    - 총 파라미터: {format_number(depth_params['total'])} ({depth_params['total']:,})")
            print(f"    - 학습 가능: {format_number(depth_params['trainable'])} ({depth_params['trainable']:,})")
            print(f"    - 고정됨: {format_number(depth_params['frozen'])} ({depth_params['frozen']:,})")
            print(f"  💾 메모리 사용량:")
            print(f"    - 모델 파라미터: {depth_memory['model_params']:.2f} MB")
            print(f"    - 모델 버퍼: {depth_memory['model_buffers']:.2f} MB")
            print(f"    - 입력 텐서: {depth_memory['input_memory']:.2f} MB")
            print(f"    - 총 메모리: {depth_memory['total']:.2f} MB")
        else:
            print("  ❌ 깊이 추정 모델 로드 실패")
            depth_params = {"total": 0, "trainable": 0, "frozen": 0}
            depth_memory = {"total": 0}
    else:
        print("  ❌ 깊이 추정 모델을 사용할 수 없습니다")
        depth_params = {"total": 0, "trainable": 0, "frozen": 0}
        depth_memory = {"total": 0}
    
    # 3. 멀티태스크 모델과 비교 (선택사항)
    print("\n🔍 멀티태스크 모델과 비교...")
    if MTL_MODEL_AVAILABLE:
        # 멀티태스크 모델은 체크포인트 없이 새로 생성
        try:
            mtl_model = LightningESANetMTL(
                height=height,
                width=width,
                num_classes=7,
                encoder_rgb='resnet34',
                encoder_depth='resnet34',
                encoder_block='NonBottleneck1D',
                use_uncertainty_weighting=True,
                use_dwa_weighting=False,
            )
            mtl_params = count_parameters(mtl_model)
            mtl_memory = estimate_memory_usage(mtl_model, (1, 3, height, width))
            
            print(f"  📊 멀티태스크 모델 파라미터 수:")
            print(f"    - 총 파라미터: {format_number(mtl_params['total'])} ({mtl_params['total']:,})")
            print(f"    - 학습 가능: {format_number(mtl_params['trainable'])} ({mtl_params['trainable']:,})")
            print(f"    - 고정됨: {format_number(mtl_params['frozen'])} ({mtl_params['frozen']:,})")
            print(f"  💾 메모리 사용량: {mtl_memory['total']:.2f} MB")
        except Exception as e:
            print(f"  ❌ 멀티태스크 모델 생성 실패: {e}")
            mtl_params = {"total": 0, "trainable": 0, "frozen": 0}
            mtl_memory = {"total": 0}
    else:
        print("  ❌ 멀티태스크 모델을 사용할 수 없습니다")
        mtl_params = {"total": 0, "trainable": 0, "frozen": 0}
        mtl_memory = {"total": 0}
    
    # 4. 순차 실행 시 총 파라미터 수 계산
    print("\n" + "="*80)
    print("순차 실행 분석 결과")
    print("="*80)
    
    sequential_total = seg_params['total'] + depth_params['total']
    sequential_trainable = seg_params['trainable'] + depth_params['trainable']
    sequential_frozen = seg_params['frozen'] + depth_params['frozen']
    sequential_memory = seg_memory['total'] + depth_memory['total']
    
    print(f"📊 순차 실행 (의미 분할 + 깊이 추정):")
    print(f"  - 총 파라미터: {format_number(sequential_total)} ({sequential_total:,})")
    print(f"  - 학습 가능: {format_number(sequential_trainable)} ({sequential_trainable:,})")
    print(f"  - 고정됨: {format_number(sequential_frozen)} ({sequential_frozen:,})")
    print(f"  - 총 메모리: {sequential_memory:.2f} MB")
    
    print(f"\n📊 멀티태스크 모델:")
    print(f"  - 총 파라미터: {format_number(mtl_params['total'])} ({mtl_params['total']:,})")
    print(f"  - 학습 가능: {format_number(mtl_params['trainable'])} ({mtl_params['trainable']:,})")
    print(f"  - 고정됨: {format_number(mtl_params['frozen'])} ({mtl_params['frozen']:,})")
    print(f"  - 총 메모리: {mtl_memory['total']:.2f} MB")
    
    # 5. 효율성 분석
    print(f"\n📈 효율성 분석:")
    if mtl_params['total'] > 0:
        param_efficiency = (sequential_total / mtl_params['total']) * 100
        memory_efficiency = (sequential_memory / mtl_memory['total']) * 100
        
        print(f"  - 파라미터 효율성: {param_efficiency:.1f}% (순차/멀티태스크)")
        print(f"  - 메모리 효율성: {memory_efficiency:.1f}% (순차/멀티태스크)")
        
        if param_efficiency > 100:
            print(f"  - 순차 실행이 멀티태스크보다 {param_efficiency-100:.1f}% 더 많은 파라미터 사용")
        else:
            print(f"  - 멀티태스크가 순차 실행보다 {100-param_efficiency:.1f}% 더 많은 파라미터 사용")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="ESANet 순차 모델 파라미터 수 측정")
    parser.add_argument("--seg_ckpt", type=str, required=True, help="의미 분할 모델 체크포인트 경로")
    parser.add_argument("--depth_ckpt", type=str, required=True, help="깊이 추정 모델 체크포인트 경로")
    parser.add_argument("--height", type=int, default=512, help="입력 이미지 높이")
    parser.add_argument("--width", type=int, default=512, help="입력 이미지 너비")
    parser.add_argument("--device", type=str, default="cpu", help="디바이스 (cpu/cuda)")
    
    args = parser.parse_args()
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.seg_ckpt):
        print(f"❌ 의미 분할 체크포인트 파일을 찾을 수 없습니다: {args.seg_ckpt}")
        return
    
    if not os.path.exists(args.depth_ckpt):
        print(f"❌ 깊이 추정 체크포인트 파일을 찾을 수 없습니다: {args.depth_ckpt}")
        return
    
    measure_sequential_models(
        seg_ckpt_path=args.seg_ckpt,
        depth_ckpt_path=args.depth_ckpt,
        height=args.height,
        width=args.width,
        device=args.device
    )


if __name__ == "__main__":
    main()
