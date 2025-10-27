#!/usr/bin/env python3
# ============================================================================
# ESANet RGB-Only Model FLOPs Measurement Script
# ============================================================================
# 이 스크립트는 ESANet RGB 전용 모델의 FLOPs를 정확하게 측정합니다.
# 주요 기능:
# - train_esanet_seg_only.py에서 실제 사용하는 ESANetRGBOnly 모델 측정
# - RGB 입력만 사용 (Depth 제거)
# - 세그멘테이션 단일 태스크 FLOPs 분석

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================================
# 실제 학습 코드에서 모델 import
# ============================================================================
try:
    # train_esanet_seg_only.py에서 실제 사용하는 모델 import
    from models.train_esanet_seg_only import ESANetRGBOnly
    ESANET_AVAILABLE = True
    print("✅ ESANetRGBOnly 모델을 성공적으로 import했습니다.")
except ImportError as e:
    print(f"❌ ESANetRGBOnly 모델 import 실패: {e}")
    ESANET_AVAILABLE = False
    sys.exit(1)

# FLOPs 측정 라이브러리
try:
    import thop
    THOP_AVAILABLE = True
    print("✅ thop를 성공적으로 import했습니다.")
except ImportError:
    print("⚠️ thop가 설치되지 않았습니다. pip install thop를 실행하세요.")
    THOP_AVAILABLE = False
    sys.exit(1)


# ============================================================================
# FLOPs Measurement Functions
# ============================================================================
def measure_flops_thop(model: nn.Module, input_rgb: torch.Tensor) -> Dict:
    """
    thop를 사용하여 FLOPs를 측정합니다 (RGB 입력만).
    
    Args:
        model: 측정할 모델
        input_rgb: RGB 입력 텐서 [B, 3, H, W]
        
    Returns:
        Dict: FLOPs 측정 결과
    """
    if not THOP_AVAILABLE:
        return {"error": "thop not available"}
    
    try:
        # 모델을 평가 모드로 설정
        model.eval()
        
        # FLOPs와 파라미터 수 측정
        with torch.no_grad():
            flops, params = thop.profile(
                model, 
                inputs=(input_rgb,), 
                verbose=False
            )
        
        return {
            "total_flops": flops,
            "total_params": params,
            "method": "thop"
        }
    except Exception as e:
        return {"error": f"thop measurement failed: {e}"}


def measure_model_parameters(model: nn.Module) -> Dict:
    """
    모델의 파라미터 수를 측정합니다.
    
    Args:
        model: 측정할 모델
        
    Returns:
        Dict: 파라미터 수 정보
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 계산 (MB)
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": size_all_mb
    }


# ============================================================================
# Main FLOPs Measurement Function
# ============================================================================
def measure_esanet_rgb_only_flops(
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    num_classes: int = 7,
    encoder_rgb: str = 'resnet34',
    encoder_block: str = 'NonBottleneck1D',
    pretrained_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    ESANet RGB 전용 모델의 FLOPs를 측정합니다.
    
    Args:
        height: 입력 이미지 높이
        width: 입력 이미지 너비
        batch_size: 배치 크기
        num_classes: 세그멘테이션 클래스 수
        encoder_rgb: RGB 인코더 백본
        encoder_block: 인코더 블록 타입
        pretrained_path: 사전훈련된 가중치 경로
        device: 계산 디바이스
        
    Returns:
        Dict: FLOPs 측정 결과
    """
    print("=" * 80)
    print("ESANet RGB-Only Model FLOPs Measurement")
    print("=" * 80)
    print(f"Input size: {height}x{width}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # 디바이스 설정
    device = torch.device(device)
    
    # 모델 초기화 (실제 학습 코드와 동일한 구조)
    print("🔧 Initializing ESANet RGB-Only Model...")
    model = ESANetRGBOnly(
        height=height,
        width=width,
        num_classes=num_classes,
        encoder_rgb=encoder_rgb,
        encoder_block=encoder_block,
        pretrained_path=pretrained_path,
    )
    
    # 모델을 디바이스로 이동
    model = model.to(device)
    
    # RGB 입력 텐서 생성 (Depth 없음!)
    print(f"📊 Creating RGB input tensor...")
    input_rgb = torch.randn(batch_size, 3, height, width, device=device)
    print(f"RGB input shape: {input_rgb.shape}")
    
    # 파라미터 수 측정
    print("\n📈 Measuring model parameters...")
    param_info = measure_model_parameters(model)
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    print(f"Model size: {param_info['model_size_mb']:.2f} MB")
    
    # FLOPs 측정
    results = {
        "model_info": {
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "encoder_rgb": encoder_rgb,
            "encoder_block": encoder_block,
            "input_modality": "RGB only",
            "device": str(device)
        },
        "parameters": param_info,
        "flops_measurements": {}
    }
    
    # thop를 사용한 FLOPs 측정
    print("\n🔍 Measuring FLOPs with thop...")
    thop_result = measure_flops_thop(model, input_rgb)
    if "error" not in thop_result:
        print(f"Total FLOPs (thop): {thop_result['total_flops']:,}")
        print(f"Total parameters (thop): {thop_result['total_params']:,}")
        results["flops_measurements"]["thop"] = thop_result
    else:
        print(f"thop measurement failed: {thop_result['error']}")
    
    return results


# ============================================================================
# Main Function
# ============================================================================
def main() -> None:
    """
    메인 함수: 512x512 입력 크기로 FLOPs 측정을 수행합니다.
    """
    print("🚀 Starting ESANet RGB-Only FLOPs Measurement...")
    print("📏 Target input size: 512x512")
    print("📝 Model: ESANetRGBOnly (실제 학습 코드와 동일)")
    
    try:
        # 실제 학습과 동일한 입력 크기로 측정
        result = measure_esanet_rgb_only_flops(
            height=512,
            width=512,
            batch_size=1,
            num_classes=7,
            encoder_rgb='resnet34',
            encoder_block='NonBottleneck1D',
            pretrained_path=None,  # 사전훈련 가중치 없이 측정
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 최종 결과 요약
        print(f"\n{'='*80}")
        print("📊 ESANet RGB-Only Model Performance Summary")
        print(f"{'='*80}")
        
        params = result["parameters"]["total_params"]
        size_mb = result["parameters"]["model_size_mb"]
        
        print(f"Model: ESANetRGBOnly (RGB 입력만)")
        print(f"Input Size: 512x512")
        print(f"Parameters: {params:,}")
        print(f"Model Size: {size_mb:.2f} MB")
        
        if "thop" in result["flops_measurements"]:
            macs = result["flops_measurements"]["thop"]["total_flops"]
            approx_flops = macs * 2  # MACs를 FLOPs로 근사 변환
            gflops = approx_flops / 1e9
            
            print(f"\nComputational Complexity:")
            print(f"Total MACs: {macs:,}")
            print(f"Approx FLOPs (~2x MACs): {approx_flops:,}")
            print(f"Note: FLOPs = MACs * 2는 근사 계산이며, 실제 FLOPs는 activation 등 추가 연산으로 인해 더 클 수 있습니다. (약 2배 더 클 수 있습니다.)")
            print(f"GFLOPs: {gflops:.2f}")
            print(f"\nPer-Pixel Metrics:")
            print(f"MACs per pixel: {macs / (512 * 512):.2f}")
            print(f"FLOPs per pixel (approx): {approx_flops / (512 * 512):.2f}")
        
        print(f"{'='*80}")
        print("✅ FLOPs measurement completed!")
        
    except Exception as e:
        print(f"❌ Error during measurement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
