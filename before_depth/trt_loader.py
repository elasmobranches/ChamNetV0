import tensorrt as trt
import numpy as np
import cv2
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class TensorRTInference:
    def __init__(self, engine_path):
        # 1. TensorRT 로거 및 런타임 생성
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        
        # 2. 엔진 로드
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("엔진 데시리얼라이즈 실패")
        self.context = self.engine.create_execution_context()
        
        # 3. TensorRT 초기화 후 torch 임포트
        import torch
        self.torch = torch
        self.stream = self.torch.cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        
        # ★ 입력 버퍼 재사용을 위한 변수
        self.input_buffer = None
        
        print(f"\n🔍 [Engine Debug] 입출력 바인딩 설정")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            
            # ★ Dynamic shape 처리 (-1 → 실제 값으로 대체)
            is_dynamic = any(d == -1 for d in shape)
            if is_dynamic:
                print(f"  ⚠️ Dynamic shape 감지: {shape}")
                for idx, dim in enumerate(shape):
                    if dim == -1:
                        shape[idx] = 1 if idx == 0 else 512  # batch=1, H/W=512
                print(f"     → 고정값으로 대체: {shape}")
                
                # Dynamic이면 context에 명시적 설정 필요
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, shape)
            
            size = int(np.prod(shape))
            if size <= 0:
                raise RuntimeError(f"잘못된 텐서 크기: {name}, shape={shape}, size={size}")
            
            gpu_mem = self.torch.zeros(
                size, 
                dtype=self._torch_dtype(trt.nptype(dtype)), 
                device='cuda'
            )
            
            self.context.set_tensor_address(name, gpu_mem.data_ptr())
            
            info = {'name': name, 'gpu': gpu_mem, 'shape': tuple(shape)}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)
            
            mem_mb = (size * 4) / (1024 * 1024)  # float32 가정
            print(f"  ✅ '{name}': shape={shape}, size={size:,} ({mem_mb:.1f}MB)")
        
        # ★ 초기화 완료 후 VRAM 상태
        allocated = self.torch.cuda.memory_allocated() / 1024**2
        print(f"\n📊 [GPU] 초기 할당: {allocated:.1f}MB")

    def _torch_dtype(self, np_dtype):
        if np_dtype == np.float32: return self.torch.float32
        if np_dtype == np.float16: return self.torch.float16
        if np_dtype == np.int32: return self.torch.int32
        if np_dtype == np.int64: return self.torch.int64
        return self.torch.float32

    def infer(self, img):
        in_info = self.inputs[0]
        h, w = in_info['shape'][2], in_info['shape'][3]
        
        # 전처리
        img_res = cv2.resize(img, (w, h))
        img_in = img_res.transpose(2, 0, 1).astype(np.float32)
        img_in = np.ascontiguousarray(np.expand_dims(img_in, axis=0))
        
        with self.torch.cuda.stream(self.stream):
            # ★ 핵심: 버퍼 재사용으로 메모리 누수 방지
            if self.input_buffer is None:
                # 최초 1회만 할당
                self.input_buffer = self.torch.empty(
                    img_in.shape, 
                    dtype=self.torch.float32, 
                    device='cuda'
                )
            
            # CPU → GPU 복사 (새 텐서 생성 없이 기존 버퍼에 덮어쓰기)
            self.input_buffer.copy_(self.torch.from_numpy(img_in))
            in_info['gpu'].copy_(self.input_buffer.view(-1))
            
            # 추론 실행
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        self.stream.synchronize()
        
        # 결과 처리
        mask_out, depth_out = None, None
        for out in self.outputs:
            res = out['gpu'].cpu().numpy().reshape(out['shape'])
            if res.shape[1] > 1:  # Segmentation
                mask_out = np.argmax(res, axis=1).squeeze().astype(np.uint8)
            elif res.shape[1] == 1:  # Depth
                depth_out = res.squeeze()
                
        return mask_out, depth_out