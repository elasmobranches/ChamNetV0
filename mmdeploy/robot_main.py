import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# ==========================================
# [설정] 사용자 모델에 맞춘 커스텀 설정
# ==========================================
ENGINE_PATH = "robot_model.engine"
INPUT_SHAPE = (512, 512)
CAMERA_ID = 0
WINDOW_NAME = "Jetson Orin - Smart Farm Monitoring"

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


CLASSES = ('background', 'chamoe', 'heatpipe', 'path', 'pillar', 'topdownfarm', 'unknown')

# 입력하신 RGB 팔레트
_PALETTE_RGB = [
    [0, 0, 0],       # background - black
    [255, 255, 0],   # chamoe - yellow
    [255, 0, 0],     # heatpipe - red
    [0, 255, 0],     # path - green
    [0, 0, 255],     # pillar - blue
    [255, 0, 255],   # topdownfarm - magenta
    [128, 128, 128]  # unknown - gray
]

# OpenCV 사용을 위해 RGB -> BGR로 자동 변환 및 포맷 맞춤
COLORS = np.array(_PALETTE_RGB, dtype=np.uint8)[:, ::-1]
# ==========================================

class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"🚀 엔진 로딩 중: {engine_path}")
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print(f"❌ 엔진 파일({engine_path})을 찾을 수 없습니다. trtexec로 먼저 변환해주세요.")
            exit(1)
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.allocations = []
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            
            size = trt.volume(shape) * 1
            dtype_np = np.float32
            
            host_mem = cuda.pagelocked_empty(size, dtype_np)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.allocations.append(int(device_mem))
            
            binding_info = {
                "index": i,
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape
            }
            
            if is_input:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, image):
        # 1. 전처리 (메모리 정렬 포함)
        img_resized = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
        img_float = img_resized.astype(np.float32)
        img_norm = (img_float - MEAN) / STD
        img_chw = img_norm.transpose((2, 0, 1))
        input_data = np.ascontiguousarray(img_chw.ravel())
        
        # 2. 추론 루틴
        np.copyto(self.inputs[0]['host'], input_data)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        # 3. 결과 정리
        results = {}
        for out in self.outputs:
            shape = out['shape']
            results[out['name']] = out['host'].reshape(shape)
            
        return results, img_resized

def draw_legend(vis_img):
    """화면 우측 하단에 클래스 범례(Legend) 그리기"""
    h, w = vis_img.shape[:2]
    legend_h = 25 * len(CLASSES) + 10
    legend_w = 150
    
    # 반투명 배경 박스
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (w - legend_w, h - legend_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0, vis_img)
    
    # 텍스트 쓰기
    for i, (name, color) in enumerate(zip(CLASSES, COLORS)):
        if i == 0: continue # 배경은 생략 가능
        # color는 BGR, 텍스트 색상은 흰색
        pt1 = (w - legend_w + 10, h - legend_h + 25 * i + 20)
        cv2.circle(vis_img, (pt1[0], pt1[1]-5), 6, color.tolist(), -1)
        cv2.putText(vis_img, name, (pt1[0] + 15, pt1[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def visualize(img, seg_prob, depth_map, fps):
    # Segmentation (Argmax -> Color Mapping)
    seg_idx = np.argmax(seg_prob[0], axis=0).astype(np.uint8)
    seg_color = COLORS[seg_idx] # 지정된 팔레트 사용
    
    # Depth (Normalize -> Jet Colormap)
    depth = depth_map[0, 0]
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # 화면 합성 (좌: Seg Overlay, 우: Depth)
    seg_overlay = cv2.addWeighted(img, 0.7, seg_color, 0.3, 0)
    
    # 구분선 그리기
    combined = np.hstack((seg_overlay, depth_color))
    cv2.line(combined, (512, 0), (512, 512), (255, 255, 255), 2)
    
    # 범례 추가
    draw_legend(combined)
    
    # 정보 텍스트
    cv2.putText(combined, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Segmentation", (15, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(combined, "Depth Estimation", (512 + 15, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return combined

def main():
    trt_wrapper = TRTWrapper(ENGINE_PATH)
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 젯슨 성능 최적화를 위해 카메라 해상도 조절 (옵션)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 카메라 연결 실패. CAMERA_ID를 확인하세요.")
        return

    print("✅ 로봇 추론 시작... (종료: 'q')")
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        # 추론
        outputs, resized_img = trt_wrapper.infer(frame)
        
        # 키값 매핑 (출력 이름 확인 필요)
        keys = list(outputs.keys())
        seg_key = next((k for k in keys if 'seg' in k), keys[0])
        depth_key = next((k for k in keys if 'depth' in k), keys[1])
        
        # 시각화 및 FPS 계산
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        final_view = visualize(resized_img, outputs[seg_key], outputs[depth_key], fps)
        
        cv2.imshow(WINDOW_NAME, final_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()