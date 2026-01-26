import sys
import os
import cv2
import time
import numpy as np
import datetime  # 날짜 모듈 추가

# [수정] 같은 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0
# 파일명은 아래에서 자동으로 생성됨

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
        print("❌ Camera Error!")
        return

    # [수정] 날짜 기반 파일명 생성
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"/data/drive_{now}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_filename, fourcc, 30.0, (1280, 720))

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
            mask = trt_model.infer(frame)
            infer_end = time.time()
            
            current_fps = 1.0 / (infer_end - infer_start)
            fps_list.append(current_fps)

            color_mask = COLORS[mask % len(COLORS)]
            color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

            cv2.putText(combined, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(combined)
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = int(time.time() - start_time)
                print(f"🔴 녹화 중... {elapsed}초 | FPS: {current_fps:.1f}", end='\r')

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