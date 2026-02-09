import sys
import os
import cv2
import time
import numpy as np
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0

COLORS = np.array([
    [0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0],
    [255, 0, 0], [255, 0, 255], [128, 128, 128]
], dtype=np.uint8)

def main():
    try:
        trt_model = TensorRTInference(ENGINE_PATH)
    except Exception as e:
        print(f"❌ 엔진 로드 실패: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Camera Error!")
        return

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"/data/drive_{now}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # Seg + Depth 합쳐서 저장 (2560x720)
    out = cv2.VideoWriter(save_filename, fourcc, 30.0, (2560, 720))

    print(f"\n🎥 녹화 시작! (Ctrl+C로 종료)")
    print(f"💾 저장 파일: {save_filename}")
    
    start_time = time.time()
    frame_count = 0
    fps_list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame.shape[1] == 2560: frame = frame[:, :1280]

            infer_start = time.time()
            mask, depth = trt_model.infer(frame)
            infer_end = time.time()
            
            current_fps = 1.0 / (infer_end - infer_start)
            fps_list.append(current_fps)

            # Segmentation 시각화
            color_mask = COLORS[mask % len(COLORS)]
            color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_view = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

            # Depth 시각화
            depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-5) * 255).astype(np.uint8)
            depth_norm = 255 - depth_norm
            depth_view = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            # 텍스트 표시
            cv2.putText(seg_view, f"FPS: {current_fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
            cv2.putText(depth_view, f"Dist: {center_depth:.2f}m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 합치기 & 저장
            combined = np.hstack((seg_view, depth_view))
            out.write(combined)
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = int(time.time() - start_time)
                print(f"🔴 녹화 중... {elapsed}초 | FPS: {current_fps:.1f} | Dist: {center_depth:.2f}m", end='\r')

    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단 (Ctrl+C)")
        
    finally:
        cap.release()
        out.release()
        if fps_list:
            print(f"\n✅ [녹화 완료]")
            print(f" - 파일: {save_filename}")
            print(f" - 평균 FPS: {np.mean(fps_list):.2f}")
            print(f" - 시간: {int(time.time() - start_time)}초")

if __name__ == "__main__":
    main()