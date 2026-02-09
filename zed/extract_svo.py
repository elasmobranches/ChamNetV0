import pyzed.sl as sl
import cv2
import sys
import time
import os
import numpy as np
from datetime import datetime
import signal

ENABLE_DISPLAY = False
zed = None
recording_active = False
shutdown_in_progress = False
svo_fullpath = None

def signal_handler(sig, frame):
    global shutdown_in_progress
    if shutdown_in_progress:
        sys.exit(0)
    shutdown_in_progress = True
    print("\n\nCtrl-C 감지. 안전하게 종료합니다...")

def main():
    global zed, recording_active, shutdown_in_progress, svo_fullpath
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # ===== Depth 설정 =====
    '''init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS'''
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_stabilization = 1
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # 밀리미터로 변경 (더 정밀)
    init_params.depth_minimum_distance = 200  # 최소 거리 20cm
    init_params.depth_maximum_distance = 20000  # 최대 거리 20m
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    
    # 제드 눈 열어!
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera open failed: {err}")
        sys.exit(1)

    # ===== 밝기 (노출/게인) 자동 조절 설정 =====
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1) # 수동 설정 값 초기화
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1) # 수동 설정 값 초기화
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, -1) # 1: 자동 모드 활성화 (Auto)
    print("✓ 밝기 조절: 자동 모드 활성화 완료")
    
    # Depth 모드 확인
    depth_mode = zed.get_init_parameters().depth_mode
    print(f"Depth Mode: {depth_mode}")
    
    cam_info = zed.get_camera_information()
    print(f"Camera Model: {cam_info.camera_model}")
    print(f"Serial Number: {cam_info.serial_number}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS: {cam_info.camera_configuration.fps}")
    
    output_dir = "/home/cv_test/Desktop/depth_recording/recordings"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svo_filename = f"zed_{timestamp}.svo2"
    svo_fullpath = os.path.join(output_dir, svo_filename)
    
    print(f"저장 경로: {svo_fullpath}\n")
    
    # SVO 녹화 시작 - depth 관련 설정 추가
    recording_params = sl.RecordingParameters()
    recording_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
    recording_params.video_filename = svo_fullpath
    recording_params.transcode_streaming_input = False  # False로 변경
    recording_params.bitrate = 0  # 자동 비트레이트
    
    ret = zed.enable_recording(recording_params)
    if ret != sl.ERROR_CODE.SUCCESS:
        print(f"Enable recording failed: {ret}")
        zed.close()
        sys.exit(1)
    
    recording_active = True
    print(f"✓ 녹화 시작: {svo_filename}")
    
    # Runtime 파라미터 - confidence 값 수정
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100  
    runtime.texture_confidence_threshold = 100 
    runtime.enable_depth = True
    runtime.enable_fill_mode = True  # Depth hole filling 활성화
    
    # Depth Mat
    depth_map = sl.Mat()
    image_left = sl.Mat()
    
    print("녹화 중... (Ctrl-C로 종료)\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time() # FPS 계산용 이전 시간
    fps_frame_counter = 0 # FPS 계산용 프레임 카운터
    first_frame_checked = False
    
    try:
        while not shutdown_in_progress:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                fps_frame_counter += 1
                
                # --- 실시간 FPS 계산 및 출력 ---
                current_time = time.time()
                time_diff = current_time - last_fps_time
                if time_diff >= 1.0: # 1초마다 출력
                    current_fps = fps_frame_counter / time_diff
                    print(f"녹화 중... | 현재 FPS: {current_fps:.1f}", end='\r')
                    last_fps_time = current_time
                    fps_frame_counter = 0
                # ------------------------------

                # 첫 프레임과 매 100프레임마다 depth 확인
                if not first_frame_checked or frame_count % 100 == 0:
                    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU)
                    depth_data = depth_map.get_data()
                    
                    # NaN과 inf 제거
                    valid_mask = ~np.isnan(depth_data) & ~np.isinf(depth_data) & (depth_data > 0)
                    valid_pixels = valid_mask.sum()
                    total_pixels = depth_data.size
                    
                    if frame_count == 1:
                        print(f"\n첫 프레임 Depth 검증:")
                    else:
                        print(f"\n\n프레임 {frame_count} Depth 상태:")
                    
                    print(f"  유효 픽셀: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
                    
                    if valid_pixels == 0:
                        print("  ⚠️ 경고: Depth 데이터가 없습니다!")
                        print("  - confidence_threshold 값을 조정해보세요")
                        print("  - 카메라가 너무 가까운 물체를 보고 있는지 확인하세요")
                    else:
                        valid_depth = depth_data[valid_mask]
                        # 밀리미터를 미터로 변환
                        print(f"  Min: {valid_depth.min()/1000:.2f}m")
                        print(f"  Max: {valid_depth.max()/1000:.2f}m")
                        print(f"  Mean: {valid_depth.mean()/1000:.2f}m")
                        print(f"  ✓ Depth 정상")
                    
                    first_frame_checked = True
                
                # 디스플레이 (ENABLE_DISPLAY True 시)
                if ENABLE_DISPLAY:
                    try:
                        zed.retrieve_image(image_left, sl.VIEW.LEFT)
                        frame_rgb = image_left.get_data()
                        
                        # FPS 텍스트 추가
                        cv2.putText(frame_rgb, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow("ZED Live", frame_rgb)
                        
                        # Depth 시각화
                        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU)
                        depth_display = depth_map.get_data()
                        depth_visual = np.nan_to_num(depth_display)
                        valid_depth = depth_visual[depth_visual > 0]
                        
                        if len(valid_depth) > 0:
                            min_d = np.percentile(valid_depth, 5)
                            max_d = np.percentile(valid_depth, 95)
                            depth_visual = np.clip(depth_visual, min_d, max_d)
                            depth_visual = ((depth_visual - min_d) / (max_d - min_d) * 255).astype(np.uint8)
                        else:
                            depth_visual = np.zeros_like(depth_visual, dtype=np.uint8)
                        
                        depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
                        cv2.imshow("Depth", depth_visual)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n사용자 요청으로 종료")
                            break
                    except cv2.error:  # cv2 관련 에러만 캐치
                        pass
                
            else:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    
    finally:
        elapsed = time.time() - start_time
        
        print(f"\n총 {frame_count} 프레임 녹화 완료")
        
        if recording_active:
            print("녹화 종료 중... (기다려주세요!)")
            # 녹화 종료 전에 마지막 프레임들이 저장되도록 대기
            time.sleep(0.5)
            ret = zed.disable_recording()
            if ret != sl.ERROR_CODE.SUCCESS and ret is not None:
                print(f"⚠️ 녹화 종료 에러: {ret}")
            else:
                print("✓ 녹화 정상 종료")
            recording_active = False
        
        if zed is not None:
            zed.close()
            print("✓ 카메라 닫기 완료")
        
        if ENABLE_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print(f"\n=== 녹화 완료 ===")
        print(f"파일: {svo_fullpath}")
        print(f"총 시간: {elapsed:.1f}초")
        print(f"평균 FPS: {frame_count/elapsed:.1f}")
        
        time.sleep(1)  # 파일 쓰기 완료 대기
        
        # 파일 체크
        if os.path.exists(svo_fullpath):
            file_size = os.path.getsize(svo_fullpath) / (1024**2)
            print(f"\n✓ 파일 저장 완료")
            print(f"  경로: {svo_fullpath}")
            print(f"  크기: {file_size:.1f} MB")
            
            if file_size < 1:
                print("  ⚠️ 파일이 너무 작습니다")
            else:
                print("  ✓ 정상")
        else:
            print(f"\n⚠️ 파일을 찾을 수 없습니다: {svo_fullpath}")

if __name__ == "__main__":
    main()