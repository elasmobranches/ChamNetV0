import sys
import os
import cv2
import time
import numpy as np
import signal
import datetime
from flask import Flask, Response

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
app = Flask(__name__)
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0

# 클래스별 색상 (BGR)

CLASS_COLORS = np.array([
    [0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0],
    [255, 0, 0], [255, 0, 255], [128, 128, 128]
], dtype=np.uint8)

# 전역 변수
model = None
cap = None
is_running = True
session_dir = ""
frame_idx = 0


def init_resources():
    global model, cap, session_dir, frame_idx
    
    # 1. 모델 로드
    if model is None:
        print("🚀 [System] AI 엔진 로드 중...")
        model = TensorRTInference(ENGINE_PATH)
        print("✅ [System] 엔진 로드 완료!")
    
    # 2. 카메라 연결
    if cap is None:
        print("📷 [System] 카메라 연결 중...")
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ [Error] 카메라 열기 실패!")
            sys.exit(1)
        print("✅ [System] 카메라 연결 성공!")
    
    # 3. 저장 폴더 생성
    if session_dir == "":
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"/data/recordings/{now}"
        os.makedirs(f"{session_dir}/rgb", exist_ok=True)
        os.makedirs(f"{session_dir}/pred_depth", exist_ok=True)
        os.makedirs(f"{session_dir}/pred_seg", exist_ok=True)
        frame_idx = 0
        print(f"📁 [Record] 저장 폴더: {session_dir}")


def cleanup(sig=None, frame=None):
    global is_running, cap, session_dir, frame_idx
    print("\n🛑 [System] 종료 신호 감지! 안전하게 종료합니다...")
    is_running = False
    time.sleep(0.3)
    
    # 메타데이터 저장
    if session_dir:
        meta = {
            'total_frames': frame_idx,
            'timestamp': datetime.datetime.now().isoformat(),
            'resolution': '1280x720'
        }
        np.save(f"{session_dir}/meta.npy", meta)
        print(f"💾 [Save] 저장 완료: {session_dir}")
        print(f"   - RGB: {frame_idx} frames")
        print(f"   - Pred Depth: {frame_idx} files")
        print(f"   - Pred Seg: {frame_idx} files")
    
    if cap:
        cap.release()
        print("📷 [System] 카메라 해제 완료")
    
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def gen_frames():
    global is_running, frame_idx
    
    init_resources()
    
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    while is_running:
        success, frame = cap.read()
        if not success:
            print("⚠️ 프레임 읽기 실패, 재시도...")
            time.sleep(0.1)
            continue
        
        # 왼쪽 눈만 사용
        raw_img = frame[:, :1280]
        
        # AI 추론
        mask, depth = model.infer(raw_img)
        
        # --- 프레임별 저장 ---
        cv2.imwrite(f"{session_dir}/rgb/{frame_idx:06d}.png", raw_img)
        
        if depth is not None:
            np.save(f"{session_dir}/pred_depth/{frame_idx:06d}.npy", depth.astype(np.float32))
        
        if mask is not None:
            np.save(f"{session_dir}/pred_seg/{frame_idx:06d}.npy", mask.astype(np.uint8))
        
        frame_idx += 1
        
        # --- Segmentation 시각화 ---
        seg_overlay = raw_img.copy()
        if mask is not None:
            mask_resized = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
            color_mask = CLASS_COLORS[mask_resized % len(CLASS_COLORS)]
            seg_overlay = cv2.addWeighted(raw_img, 0.6, color_mask, 0.4, 0)
        
        # --- Depth 시각화 ---
        depth_view = np.zeros_like(raw_img)
        if depth is not None:
            depth_resized = cv2.resize(depth, (1280, 720))
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-5) * 255).astype(np.uint8)
            depth_norm = 255 - depth_norm
            depth_view = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # FPS 계산
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_time = time.time()
            fps_count = 0
        
        # 정보 표시
        cv2.putText(seg_overlay, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if depth is not None:
            center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
            cv2.putText(depth_view, f'Dist: {center_depth:.2f}m', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 합치기
        combined = np.hstack((seg_overlay, depth_view))
        
        # 웹 전송
        combined_small = cv2.resize(combined, (1280, 360))
        ret, buffer = cv2.imencode('.jpg', combined_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    folder_name = os.path.basename(session_dir) if session_dir else "Ready..."
    return f"""
    <html>
    <head><title>Robot Vision</title></head>
    <body style="margin:0; background:#000; text-align:center;">
        <h2 style="color:#fff;">🤖 Robot Vision (Seg + Depth)</h2>
        <h3 style="color:red;">🔴 REC: {folder_name}</h3>
        <img src='/video_feed' style="width:100%; max-width:1280px;">
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("⏳ [Pre-load] 서버 시작 전 리소스 초기화...")
    init_resources()
    
    print(f"\n🌐 [Server] http://0.0.0.0:5000")
    print("   (Ctrl+C로 안전 종료 + 데이터 저장)")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
