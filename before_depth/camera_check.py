import sys
import os
import cv2
import numpy as np

# --- trt_loader 불러오기 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mtl_dir = os.path.abspath(os.path.join(current_dir, "../mtl"))
sys.path.append(mtl_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0
SAVE_PATH = "/data/check_result.jpg"

COLORS = np.array([
    [0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0],
    [255, 0, 0], [255, 0, 255], [128, 128, 128]
], dtype=np.uint8)

def main():
    try:
        trt_model = TensorRTInference(ENGINE_PATH)
    except Exception:
        return

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera Error")
        return

    print("\n=== 🎯 클래스 확인 모드 (5초간 실행) ===")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Left Eye Crop
            h, w, _ = frame.shape
            if w == 2560: frame = frame[:, :1280]
            
            # 고속 추론
            mask = trt_model.infer(frame)
            
            # 중앙 픽셀 클래스 확인
            center_y, center_x = 256, 256
            detected_class = mask[center_y, center_x]
            
            # 시각화
            if frame_count % 30 == 0:
                colored_mask = COLORS[mask % len(COLORS)]
                colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                combined = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
                
                # 십자선 그리기
                cw, ch = combined.shape[1]//2, combined.shape[0]//2
                cv2.line(combined, (cw-20, ch), (cw+20, ch), (0, 0, 255), 2)
                cv2.line(combined, (cw, ch-20), (cw, ch+20), (0, 0, 255), 2)
                
                # 텍스트
                msg = f"Center Class: {detected_class}"
                cv2.putText(combined, msg, (cw-100, ch-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imwrite(SAVE_PATH, combined)
                print(f"📸 Saved: {SAVE_PATH} (중앙 물체: {detected_class}번)")
            
            frame_count += 1
            if frame_count >= 150: break # 5초 후 종료

    finally:
        cap.release()

if __name__ == "__main__":
    main()