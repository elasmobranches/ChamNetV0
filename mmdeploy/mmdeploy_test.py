from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
import cv2
import numpy as np
import os
import tensorrt as trt # TensorRT 정보 확인용

# 1. 설정 및 경로
deploy_cfg_path = 'mmdeploy/configs/mmseg/mtl_deploy_cfg.py'
model_cfg_path = 'deploy_model_cfg.py'
device = 'cuda:0'
backend_model_path = ['mtl_onnx_out/end2end.engine']
image_path = 'dataset/test/images/20250526_rfv4_frame_000298_00m_09s.jpg'

# 2. 모델 로드
deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model_path)

# 3. 데이터 준비 (512x512 고정)
image = cv2.imread(image_path)
target_h, target_w = 512, 512
image_resized = cv2.resize(image, (target_w, target_h))
model_inputs, _ = task_processor.create_input(image_resized, input_shape=(target_w, target_h))

# 4. 추론
with torch.no_grad():
    inputs = model_inputs['inputs']
    if isinstance(inputs, list): inputs = inputs[0]
    if inputs.dim() == 3: inputs = inputs.unsqueeze(0)
    
    # [핵심] float32 변환 + GPU 이동 + Contiguous 정렬 + Clone(메모리 새로 할당)
    # 이 조합은 가장 안전한 입력 데이터를 만듭니다.
    inputs = inputs.float().to(device).contiguous().clone()
    
    print("-" * 30)
    print(f"🚀 입력 텐서 정보: {inputs.shape}, dtype={inputs.dtype}")
    print(f"🚀 메모리 정렬됨: {inputs.is_contiguous()}")
    
    # 엔진 바인딩 정보 확인 (디버깅용)
    # 엔진이 실제로 어떤 크기의 입력을 기다리는지 확인합니다.
    try:
        engine = model.wrapper.engine
        print(f"🔧 엔진 바인딩 개수: {engine.num_bindings}")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            print(f"   [{i}] {'Input ' if is_input else 'Output'}: {name}, Shape={shape}, Type={dtype}")
    except Exception as e:
        print(f"   (엔진 정보 읽기 실패: {e})")
    print("-" * 30)

    # 엔진 실행
    raw_outputs = model.wrapper({model.input_name: inputs})
    torch.cuda.synchronize()

# 5. 결과 파싱 (이전과 동일)
try:
    seg_raw = raw_outputs['seg_logits'].cpu().numpy().squeeze()
    depth_raw = raw_outputs['depth_map'].cpu().numpy().squeeze()
    seg_map = np.argmax(seg_raw, axis=0) if seg_raw.ndim == 3 else seg_raw
    
    print(f"✅ 추론 성공!")
    print(f"📊 Seg 클래스: {np.unique(seg_map)}")
    
    os.makedirs('inference_results', exist_ok=True)
    cv2.imwrite('inference_results/final_seg.png', (seg_map * 40).astype(np.uint8))
    depth_vis = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite('inference_results/final_depth.png', cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))
    print("결과 저장 완료.")

except Exception as e:
    print(f"❌ 결과 처리 중 에러: {e}")