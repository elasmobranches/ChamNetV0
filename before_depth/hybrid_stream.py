import sys
import os
import cv2
import time
import numpy as np
import signal
import datetime
from flask import Flask, Response, render_template_string

# [수정됨] 모든 파일이 같은 폴더에 있으므로 현재 폴더를 경로에 추가
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

# 전역 변수
trt_model = None
cap = None
video_writer = None
is_running = True
current_filename = ""

def init_resources():
    global trt_model, cap, video_writer, current_filename
    
    # 1. 모델 로드 (이미 로드되어 있으면 패스)
    if trt_model is None:
        print(f"🚀 [System] AI 엔진 로드 시작... (10~20초 소요될 수 있음)")
        trt_model = TensorRTInference(ENGINE_PATH)
        print("✅ [System] 엔진 로드 완료!")

    # 2. 카메라 연결
    if cap is None:
        print(f"📷 [System] 카메라 연결 중...")
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("❌ [Error] 카메라를 열 수 없습니다!")
            sys.exit(1)
        print("✅ [System] 카메라 연결 성공!")
    
    # 3. 현재 시간으로 녹화 파일 생성
    if video_writer is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_filename = f"/data/record_{now}.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # 녹화 해상도는 왼쪽 눈 크기 (1280x720)
        video_writer = cv2.VideoWriter(current_filename, fourcc, 30.0, (1280, 720))
        print(f"🔴 [Record] 녹화 시작: {current_filename}")

def cleanup(sig=None, frame=None):
    global is_running, cap, video_writer
    print("\n⏹️ [System] 종료 신호 감지! 마무리 중...")
    is_running = False
    time.sleep(0.5) # 쓰기 완료 대기
    
    if video_writer: video_writer.release()
    if cap: cap.release()
    
    if current_filename:
        print(f"💾 [Save] 파일 저장 완료: {current_filename}")
    else:
        print("⚠️ [Save] 저장된 파일 없음")
        
    sys.exit(0)

# Ctrl+C 감지
signal.signal(signal.SIGINT, cleanup)

def generate_frames():
    # 접속 시 리소스가 없으면 초기화
    init_resources()
    
    while is_running:
        success, frame = cap.read()
        if not success: break
            
        # Left Eye Crop
        h, w, _ = frame.shape
        if w == 2560: frame = frame[:, :1280]

        start = time.time()
        mask = trt_model.infer(frame)
        fps = 1.0 / (time.time() - start)

        color_mask = COLORS[mask % len(COLORS)]
        color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 파일 저장
        if video_writer:
            video_writer.write(combined)
        
        # 웹 전송
        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(f'''
        <body style="background:black; text-align:center; color:white;">
            <h1>🎥 Hybrid Stream</h1>
            <h3 style="color:red;">● REC: {os.path.basename(current_filename) if current_filename else "Ready..."}</h3>
            <img src="/video_feed" style="width:90%; border:2px solid red;">
        </body>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # [중요] 서버 시작 전에 미리 엔진을 로드해서 접속 지연 방지
    print("⏳ [Pre-load] 서버 시작 전 AI 모델을 미리 로드합니다...")
    init_resources()
    
    print(f"\n🌍 [Server] 웹 서버 시작: http://0.0.0.0:5000")
    print(f"   (브라우저에서 접속하면 화면이 보입니다)")
    
    # threaded=True로 변경하여 다중 접속 시 멈춤 현상 완화
    # (trt_loader가 내부적으로 context를 잘 잡고 있다면 True도 가능하지만, 
    #  혹시 에러나면 다시 False로 해야 함. 우선 멈춤 해결을 위해 True 시도 권장하되, 안전하게 False 유지)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)