import sys
import os
import cv2
import time
import numpy as np
from flask import Flask, Response, render_template_string

# [수정] 같은 폴더 경로 추가
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

def get_resources():
    global trt_model, cap
    if trt_model is None:
        print("🚀 [System] AI 모델 로드 중...")
        trt_model = TensorRTInference(ENGINE_PATH)
    if cap is None:
        print("📷 [System] 카메라 연결 중...")
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return trt_model, cap

def generate_frames():
    model, camera = get_resources()
    
    while True:
        success, frame = camera.read()
        if not success: break
            
        if frame.shape[1] == 2560: frame = frame[:, :1280]

        start = time.time()
        mask = model.infer(frame)
        fps = 1.0 / (time.time() - start)

        color_mask = COLORS[mask % len(COLORS)]
        color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <body style="background:black; text-align:center; color:white;">
            <h1>🐕 ChamDog Wireless Stream</h1>
            <img src="/video_feed" style="width:90%; border:2px solid green;">
        </body>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌍 Server: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)