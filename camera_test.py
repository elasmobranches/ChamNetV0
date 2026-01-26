import sys
import os
import time
import cv2
import numpy as np

# --- [핵심] 옆 폴더(mtl)에 있는 trt_loader 불러오기 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 이제 import 가능!
from trt_loader import TensorRTInference

# --- 설정 ---
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0  # ZED 카메라 (보통 0번)
SAVE_PATH = "/data/result_cam.jpg"

# 색상표 (BGR 순서, 0번 검정, 1번 노랑, 2번 빨강...)
COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [0, 255, 255],   # 1: Chamoe (Yellow)
    [0, 0, 255],     # 2: Heatpipe (Red)
    [0, 255, 0],     # 3: Path (Green)
    [255, 0, 0],     # 4: Pillar (Blue)
    [255, 0, 255],   # 5: Topdown (Magenta)
    [128, 128, 128]  # 6: Unknown
], dtype=np.uint8)

def main():
    # 1. 고속 엔진 로드 (trt_loader 사용)
    try:
        trt_model = TensorRTInference(ENGINE_PATH)
    except Exception as e:
        print(f"❌ 엔진 로드 실패: {e}")
        return

    # 2. 카메라 연결
    print(f"📷 Opening ZED Camera {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # ZED 2i 해상도 강제 설정 (2560x720 = Side by Side)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없습니다! (/dev/video{CAMERA_ID} 확인 필요)")
        return

    print("✅ Camera & Engine Ready! Press Ctrl+C to stop.")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                break
            
            # --- [전처리] ZED 이미지 자르기 (Left Eye) ---
            # 입력이 2560x720이면 반으로 잘라 1280x720 사용
            height, width, _ = frame.shape
            if width == 2560:
                frame = frame[:, :1280] 
            
            # --- [추론] 고속 추론 (166 FPS급) ---
            start = time.time()
            
            # infer 함수가 이미 GPU Argmax가 끝난 (512, 512) 마스크를 줍니다.
            mask, depth = trt_model.infer(frame)
            
            end = time.time()
            inference_fps = 1.0 / (end - start)
            
            frame_count += 1
            print(f"Frame {frame_count} | Inference FPS: {inference_fps:.1f}")

            # --- [시각화] (10프레임마다 저장) ---
            if frame_count % 10 == 0:
                # 마스크에 색칠하기
                colored_mask = COLORS[mask % len(COLORS)]
                
                # 원본 크기로 복원 (Overlay를 위해)
                colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # 반투명 합성
                combined = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
                
                # FPS 출력
                cv2.putText(combined, f"Inf FPS: {inference_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 저장
                cv2.imwrite(SAVE_PATH, combined)
                print(f"💾 Saved visualization to {SAVE_PATH}")
            
            # 100프레임만 찍고 종료 (테스트용)
            if frame_count >= 100:
                print("🏁 Test finished (100 frames).")
                break

    except KeyboardInterrupt:
        print("\n🛑 중단됨 (User Interrupt)")
    finally:
        cap.release()
        print("Camera released.")

if __name__ == "__main__":
    main()