# 파일명: /data/test/mtl/trt_loader.py
import tensorrt as trt
import numpy as np
import cv2
import torch 

class TensorRTInference:
    def __init__(self, engine_path):
        # 로거 설정
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading Engine from {engine_path}...")
        
        # 1. 엔진 로드
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("엔진 로드 실패!")

        self.context = self.engine.create_execution_context()
        
        # PyTorch Stream 사용
        self.stream = torch.cuda.Stream()

        # 메모리 보관함
        self.inputs = []
        self.outputs = []
        self.bindings = []

        # 2. 입출력 텐서 설정 (All PyTorch)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            
            # 동적 배치 대응
            if shape[0] == -1: shape = (1,) + shape[1:]
            
            # 타입 매핑
            trt_type = self.engine.get_tensor_dtype(name)
            if trt_type == trt.float32: dtype = torch.float32
            elif trt_type == trt.int32: dtype = torch.int32
            else: dtype = torch.float32

            # GPU 메모리 할당
            tensor = torch.zeros(tuple(shape), dtype=dtype, device='cuda')
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)

        print("✅ Engine Loaded & PyTorch Mode Ready!")

    def infer(self, image):
        # 입력 사이즈 (모델에 맞게 수정 필요하면 여기서)
        input_shape = (512, 512) 
        
        # 1. 전처리 (CPU -> GPU)
        resized = cv2.resize(image, input_shape)
        
        # 이미지 데이터를 PyTorch 텐서로 변환
        input_tensor_cpu = torch.from_numpy(
            np.ascontiguousarray(
                resized.transpose((2, 0, 1)).astype(np.float32)
            )
        )
        
        # GPU로 비동기 복사
        self.inputs[0].copy_(input_tensor_cpu, non_blocking=True)

        # 2. 추론 (TensorRT)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # 3. 후처리 (GPU Argmax) - Segmentation Mask
        # (1, 7, 512, 512) -> (1, 512, 512)
        seg_output = self.outputs[0]
        final_mask_gpu = torch.argmax(seg_output, dim=1).to(torch.uint8)
        
        # 결과 CPU 이동
        final_mask_cpu = final_mask_gpu.cpu().numpy()[0]
        
        return final_mask_cpu