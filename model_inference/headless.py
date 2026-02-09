import sys
import os
import cv2
import time
import numpy as np
import signal  # [핵심] 안전 종료를 위한 모듈

# --- 경로 설정: 현재 폴더(/data/test)를 라이브러리 경로에 추가 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trt_loader import TensorRTInference

# --- 설정 ---
ENGINE_PATH = "/data/model.engine"

# 전역 변수 (안전한 종료를 위해 필요)
is_running = True
cap = None

def signal_handler(sig, frame):
    """Ctrl+C가 눌리면 무조건 실행되어 안전하게 종료를 유도함"""
    global is_running
    print("\n🛑 [System] 강제 종료 신호 감지! 카메라를 안전하게 끕니다...")
    is_running = False

# 운영체제의 종료 신호(SIGINT)를 가로챔
signal.signal(signal.SIGINT, signal_handler)

def find_camera():
    """0번부터 9번까지 뒤져서 사용 가능한 ZED 카메라를 찾는 함수"""
    print("🔍 [System] 사용 가능한 카메라 포트를 탐색합니다...")
    for index in range(10):
        # cv2.CAP_V4L2 옵션은 리눅스에서 필수적임
        temp_cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if temp_cap.isOpened():
            # ZED 2i 해상도 설정 시도 (Side-by-Side 2560x720)
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 실제 영상이 들어오는지 테스트
            ret, _ = temp_cap.read()
            if ret:
                print(f"✅ [System] 카메라 발견! (Index: {index})")
                return temp_cap
            else:
                temp_cap.release()
    return None

def main():
    global cap, is_running
    
    print("🚀 [System] AI 엔진 로드 중...")
    try:
        model = TensorRTInference(ENGINE_PATH)
        print("✅ [System] 엔진 로드 완료!")
    except Exception as e:
        print(f"❌ [Error] 엔진 로드 실패: {e}")
        return

    # 스마트하게 카메라 찾기
    cap = find_camera()
    
    if cap is None:
        print("❌ [Error] 모든 카메라 포트가 응답하지 않습니다.")
        print("   👉 해결책: USB 선을 뺐다 다시 꼽거나 'sudo udevadm trigger'를 입력하세요.")
        return

    print("🤖 순수 추론 루프 시작! (Ctrl+C로 종료)")
    print("   - Infer FPS : GPU 순수 연산 속도")
    print("   - Loop FPS  : 카메라 입력 포함 전체 속도")
    print("   - Center Depth: 화면 중앙의 거리 (미터 단위 예상)")

    prev_time = time.time()
    frame_count = 0
    
    # 메인 루프
    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임 읽기 실패 (카메라 연결 끊김?)")
            break
        
        # [전처리] 왼쪽 눈 영상만 잘라내기
        if frame.shape[1] == 2560:
            frame = frame[:, :1280]

        # [추론] GPU 연산 시작
        t0 = time.time()
        
        # ★ 중요: 이제 결과가 2개(Mask, Depth) 나옵니다 ★
        # (만약 trt_loader를 수정 안 했으면 여기서 에러가 날 수 있음)
        try:
            mask, depth = model.infer(frame)
        except ValueError:
            # 혹시 trt_loader가 아직 수정 안 되어서 값 1개만 주면 호환성 유지
            result = model.infer(frame)
            if isinstance(result, tuple):
                mask, depth = result
            else:
                mask = result
                depth = None

        t1 = time.time()
        inference_fps = 1.0 / (t1 - t0)
        
        # [데이터 확인] 화면 중앙점(좌표 256, 256)의 거리 찍어보기
        center_dist = 0.0
        if depth is not None:
            # Depth 맵 크기가 512x512라고 가정
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            center_dist = depth[cy, cx]

        # [FPS 측정] 전체 루프 속도
        frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time
        
        # 1초마다 상태 리포트 출력
        if elapsed > 1.0:
            loop_fps = frame_count / elapsed
            
            # 거리 정보가 있으면 같이 출력, 없으면 FPS만 출력
            depth_info = f" | 📏 중앙 거리: {center_dist:.2f}" if depth is not None else " | 📏 Depth 없음"
            
            print(f"⚡ GPU: {inference_fps:.1f} FPS | 🎥 실제: {loop_fps:.1f} FPS{depth_info}", end='\r')
            
            prev_time = curr_time
            frame_count = 0

    # 종료 처리
    if cap:
        cap.release()
    print("\n👋 [System] 시스템이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()