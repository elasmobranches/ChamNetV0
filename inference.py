import tensorrt as trt
import numpy as np
import cv2
import time
import torch 

# --- 설정 ---
ENGINE_PATH = "model.engine"
INPUT_SHAPE = (512, 512)
LOOP_COUNT = 1000

# 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    def __init__(self, engine_path):
        print(f"Loading Engine from {engine_path}...")
        
        # 1. 엔진 로드
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("엔진 로드 실패!")

        self.context = self.engine.create_execution_context()
        
        # [핵심 변경 1] PyCUDA Stream 대신 PyTorch Stream 사용
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
            
            # TensorRT 타입을 PyTorch 타입으로 매핑
            # (대부분 Float32 아니면 Int32일 것입니다)
            trt_type = self.engine.get_tensor_dtype(name)
            if trt_type == trt.float32:
                dtype = torch.float32
            elif trt_type == trt.int32:
                dtype = torch.int32
            else:
                dtype = torch.float32 # fallback

            # [핵심 변경 2] 입력, 출력 모두 PyTorch로 GPU 메모리 할당
            # (PyCUDA 안 씀 -> 충돌 해결)
            tensor = torch.zeros(tuple(shape), dtype=dtype, device='cuda')
            
            # TensorRT에게 주소 알려주기
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
                print(f"Input Ready (Torch): {name}, Shape: {shape}")
            else:
                self.outputs.append(tensor)
                print(f"Output Ready (Torch): {name}, Shape: {shape}")

        print("Engine Loaded & Pure PyTorch Mode Ready!")

    def infer(self, image):
        # 1. 전처리 (CPU -> GPU)
        resized = cv2.resize(image, INPUT_SHAPE)
        
        # 이미지 데이터를 PyTorch 텐서로 변환 (CPU)
        input_tensor_cpu = torch.from_numpy(
            np.ascontiguousarray(
                resized.transpose((2, 0, 1)).astype(np.float32)
            )
        )
        
        # [핵심 변경 3] GPU로 비동기 복사 (Non-blocking)
        # self.inputs[0]은 이미 GPU에 있는 텐서임
        self.inputs[0].copy_(input_tensor_cpu, non_blocking=True)

        # 2. 추론 (TensorRT)
        # PyTorch가 관리하는 스트림 주소를 넘겨줌
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # 3. 후처리 (GPU Argmax)
        # 이미 self.outputs[0]에 결과가 들어있음 (Zero-copy)
        seg_output = self.outputs[0]
        
        # GPU Argmax
        final_mask_gpu = torch.argmax(seg_output, dim=1).to(torch.uint8)
        
        # 결과만 CPU로 가져오기 (이 시점에서 동기화됨)
        final_mask_cpu = final_mask_gpu.cpu().numpy()[0]
        
        return final_mask_cpu

# --- 메인 실행 ---
if __name__ == "__main__":
    try:
        # PyTorch CUDA 초기화 (필수)
        torch.cuda.init()
        
        trt_model = TensorRTInference(ENGINE_PATH)

        # 더미 이미지
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)

        print("\n--- 1. Warm up ---")
        for _ in range(10):
            trt_model.infer(dummy_image)
        print("Warm up 완료.\n")

        print(f"--- 2. Speed Test ({LOOP_COUNT} frames) ---")
        
        total_time = 0
        
        # GPU 동기화 (정확한 시간 측정을 위해)
        torch.cuda.synchronize()
        
        for i in range(LOOP_COUNT):
            loop_start = time.time()
            
            mask = trt_model.infer(dummy_image)
            
            # 루프 끝날 때마다 동기화 할 필요는 없지만, 
            # infer 함수 마지막의 .cpu()가 암시적으로 동기화를 수행함.
            
            loop_end = time.time()
            duration = loop_end - loop_start
            total_time += duration

            if i % 100 == 0:
                print(f"Frame {i}/{LOOP_COUNT} | FPS: {1.0/duration:.1f}")

        avg_fps = LOOP_COUNT / total_time
        print(f"\n[Final Clean Result]")
        print(f"Average FPS: {avg_fps:.2f} 🚀")
        print(f"Average Latency: {(total_time/LOOP_COUNT)*1000:.2f} ms")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Error] {e}")