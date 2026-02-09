import sys
import os
import cv2
import numpy as np

# --- trt_loader 불러오기 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Camera Error")
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
            
            # 추론 (Seg + Depth)
            mask, depth = trt_model.infer(frame)
            
            # 중앙 픽셀 정보
            center_y, center_x = 256, 256
            detected_class = mask[center_y, center_x]
            center_depth = depth[center_y, center_x] if depth is not None else 0.0
            
            # 시각화 (30프레임마다)
            if frame_count % 30 == 0:
                # Seg 시각화
                colored_mask = COLORS[mask % len(COLORS)]
                colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                seg_view = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
                
                # Depth 시각화
                depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
                d_min, d_max = depth_resized.min(), depth_resized.max()
                depth_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-5) * 255).astype(np.uint8)
                depth_norm = 255 - depth_norm
                depth_view = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                
                # 십자선 그리기 (Seg)
                cw, ch = seg_view.shape[1]//2, seg_view.shape[0]//2
                cv2.line(seg_view, (cw-20, ch), (cw+20, ch), (0, 0, 255), 2)
                cv2.line(seg_view, (cw, ch-20), (cw, ch+20), (0, 0, 255), 2)
                
                # 텍스트
                cv2.putText(seg_view, f"Class: {detected_class}", (cw-80, ch-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(depth_view, f"Dist: {center_depth:.2f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 합치기 & 저장
                combined = np.hstack((seg_view, depth_view))
                cv2.imwrite(SAVE_PATH, combined)
                print(f"📸 Saved: {SAVE_PATH} (Class: {detected_class}, Depth: {center_depth:.2f}m)")
            
            frame_count += 1
            if frame_count >= 150: break

    finally:
        cap.release()
        print("📷 Camera released.")

if __name__ == "__main__":
    main()