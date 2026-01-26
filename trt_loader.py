import torch
torch.zeros(1, device='cuda')  # CUDA 컨텍스트 선점

import tensorrt as trt
import numpy as np
import cv2

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"Loading Engine from {engine_path}...")
        
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("엔진 로드 실패!")
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            
            if shape[0] == -1:
                shape[0] = 1
            
            trt_dtype = self.engine.get_tensor_dtype(name)
            dtype = torch.float32 if trt_dtype == trt.float32 else torch.int32
            
            tensor = torch.zeros(shape, dtype=dtype, device='cuda')
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            info = {'name': name, 'tensor': tensor, 'shape': tuple(shape)}
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(info)
                print(f"  INPUT '{name}': {shape}")
            else:
                self.outputs.append(info)
                print(f"  OUTPUT '{name}': {shape}")
        
        self.input_buffer = None
        print("✅ Engine Loaded!")

    def infer(self, image):
        h, w = 512, 512
        
        resized = cv2.resize(image, (w, h))
        img_in = resized.transpose((2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        
        if self.input_buffer is None:
            self.input_buffer = torch.empty((1, 3, h, w), dtype=torch.float32, device='cuda')
        
        self.input_buffer.copy_(torch.from_numpy(img_in))
        self.inputs[0]['tensor'].copy_(self.input_buffer)
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        mask_out, depth_out = None, None
        
        for out in self.outputs:
            name = out['name']
            data = out['tensor']
            
            if 'seg' in name:
                mask_out = torch.argmax(data, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            elif 'depth' in name:
                depth_out = data.squeeze().cpu().numpy()
        
        return mask_out, depth_out