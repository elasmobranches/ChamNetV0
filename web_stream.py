import sys
import os
import cv2
import time
import numpy as np
import signal
from flask import Flask, Response

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
app = Flask(__name__)
ENGINE_PATH = "/data/model.engine"
CAMERA_ID = 0

COLORS = np.array([
    [0, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0],
    [255, 0, 0], [255, 0, 255], [128, 128, 128]
], dtype=np.uint8)

trt_model = None
cap = None
is_running = True


def cleanup(sig=None, frame=None):
    global is_running, cap
    print("\n🛑 [System] 종료 중...")
    is_running = False
    if cap:
        cap.release()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def get_resources():
    global trt_model, cap
    if trt_model is None:
        print("🚀 [System] AI 모델 로드 중...")
        trt_model = TensorRTInference(ENGINE_PATH)
    if cap is None:
        print("📷 [System] 카메라 연결 중...")
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return trt_model, cap


def generate_frames():
    global is_running
    model, camera = get_resources()
    
    while is_running:
        success, frame = camera.read()
        if not success: 
            time.sleep(0.1)
            continue
            
        if frame.shape[1] == 2560: frame = frame[:, :1280]

        start = time.time()
        mask, depth = model.infer(frame)
        fps = 1.0 / (time.time() - start)

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
        cv2.putText(seg_view, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        cv2.putText(depth_view, f"Dist: {center_depth:.2f}m", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 합치기
        combined = np.hstack((seg_view, depth_view))
        combined_small = cv2.resize(combined, (1280, 360))
        
        ret, buffer = cv2.imencode('.jpg', combined_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return """
    <html>
    <head><title>Robot Vision</title></head>
    <body style="margin:0; background:#000; text-align:center;">
        <h2 style="color:#fff;">🤖 Robot Vision (Seg + Depth)</h2>
        <img src='/video_feed' style="width:100%; max-width:1280px;">
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("🌐 [Server] http://0.0.0.0:5000")
    print("   (Ctrl+C로 안전 종료)")
    get_resources()  # 미리 로드
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)