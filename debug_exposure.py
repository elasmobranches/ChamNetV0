#!/usr/bin/env python3
"""
ZED 카메라 노출 디버깅 스크립트 (웹 버전)
========================================
Auto Exposure 문제 진단을 위한 웹 기반 디버깅 도구

문제 상황:
    - ZED 카메라의 Auto Exposure가 전체 화면 밝기를 기준으로 동작
    - 특정 조명 환경에서 ArUco 마커 인식률이 낮아질 수 있음
    - 마커가 화면의 작은 부분만 차지하면 노출이 부적절하게 조정됨

해결 방법:
    - 이 도구로 최적의 노출/게인/밝기/대비/선명도 값 찾기
    - Manual 모드로 전환하여 수동으로 조정
    - 마커 검출률 90% 이상 달성 시 설정값 저장
    - zed_camera.py의 open() 메서드에 최적값 하드코딩

주요 기능:
1. 실시간 프레임 밝기 모니터링 (mean, std)
2. 현재 노출/게인 값 실시간 표시
3. 웹 UI로 5가지 파라미터 수동 조정 (노출, 게인, 밝기, 대비, 선명도)
4. 마커 인식 성공/실패 통계 (검출률 %)
5. 실시간 MJPEG 스트리밍

사용법:
    python3 debug_exposure.py
    브라우저에서 http://0.0.0.0:5001 접속

워크플로우:
    1. Auto 모드로 시작 → 마커 검출률 확인
    2. Manual 모드로 전환
    3. 슬라이더로 노출/게인 조정 (목표: 검출률 90% 이상)
    4. "설정 저장" 버튼 클릭
    5. 터미널 로그의 권장값을 zed_camera.py에 적용
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from threading import Thread, Event, Lock
import time
import signal
from flask import Flask, render_template, Response, jsonify, request

# 프로젝트 모듈 임포트
from app.utils.config import Config, ConfigError
from app.utils.logger import setup_logger, get_logger
from app.camera import ZEDCamera, ZEDCameraError
from app.marker import ArucoDetector


class ExposureDebuggerWeb:
    """
    웹 기반 노출 디버깅 도구

    ZED 카메라의 노출(Exposure)과 게인(Gain) 설정을 실시간으로 조정하며
    ArUco 마커 검출률을 모니터링하는 Flask 웹 애플리케이션.

    구조:
        - Flask 서버: 웹 UI 제공 및 API 엔드포인트
        - 카메라 스레드: 백그라운드에서 프레임 캡처 및 분석
        - 실시간 통신: 500ms 간격으로 상태 업데이트

    주요 속성:
        auto_exposure (bool): True면 Auto Exposure, False면 Manual
        manual_exposure (int): Manual 모드에서 사용할 노출값 (0-100)
        manual_gain (int): Manual 모드에서 사용할 게인값 (0-100)
        manual_brightness (int): 밝기 설정 (0-8)
        manual_contrast (int): 대비 설정 (0-8)
        manual_sharpness (int): 선명도 설정 (0-8)
        frame_count (int): 총 프레임 수
        detection_count (int): 마커 검출 성공 횟수
    """

    def __init__(self, config_path: str = None, port: int = 5001):
        # 설정 로드
        try:
            self.config = Config(config_path)
        except ConfigError as e:
            print(f"❌ 설정 로드 실패: {e}")
            sys.exit(1)

        # 로거 초기화
        self.logger = setup_logger(
            name='exposure_debugger',
            log_dir=self.config.log_dir,
            level='INFO'
        )

        # 컴포넌트 초기화
        self.camera = ZEDCamera(self.config)
        self.detector = ArucoDetector(self.config)

        # Flask 앱
        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        self.port = port

        # 상태 변수
        self.auto_exposure = True
        self.manual_exposure = 50
        self.manual_gain = 50
        self.manual_brightness = 4  # 0-8, 기본값 4
        self.manual_contrast = 4    # 0-8, 기본값 4
        self.manual_sharpness = 4   # 0-8, 기본값 4

        # 통계 누적
        self.brightness_history = []
        self.exposure_history = []
        self.gain_history = []
        self.detection_count = 0
        self.frame_count = 0

        # 스레드 관련
        self.running = Event()
        self.state_lock = Lock()
        self.current_frame = None
        self.current_stats = {}
        self.shutdown_requested = Event()

        # 라우트 등록
        self._register_routes()

        # 종료 시그널 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """종료 시그널 핸들러"""
        self.logger.info("종료 시그널 수신")
        self.shutdown()

    def _register_routes(self):
        """Flask 라우트 등록"""

        @self.app.route('/')
        def index():
            """디버깅 메인 페이지"""
            return render_template('debug.html')

        @self.app.route('/video_feed')
        def video_feed():
            """비디오 스트림"""
            return Response(
                self._generate_stream(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/api/status')
        def get_status():
            """현재 상태 조회"""
            with self.state_lock:
                return jsonify(self.current_stats)

        @self.app.route('/api/toggle_auto', methods=['POST'])
        def toggle_auto():
            """Auto Exposure 토글"""
            try:
                self.auto_exposure = not self.auto_exposure
                if self.auto_exposure:
                    self.camera.set_auto_exposure()
                    self.logger.info("✓ Auto Exposure 활성화")
                    mode = 'auto'
                else:
                    self.camera.set_manual_exposure(self.manual_exposure, self.manual_gain)
                    self.logger.info(f"✓ Manual Exposure 활성화: E={self.manual_exposure}, G={self.manual_gain}")
                    mode = 'manual'

                return jsonify({'success': True, 'mode': mode})
            except Exception as e:
                self.logger.error(f"Auto Exposure 토글 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_exposure', methods=['POST'])
        def set_exposure():
            """노출 값 설정"""
            try:
                data = request.json
                exposure = int(data.get('exposure', 50))
                exposure = max(0, min(100, exposure))

                self.manual_exposure = exposure
                if not self.auto_exposure:
                    self.camera.set_manual_exposure(self.manual_exposure, self.manual_gain)
                    self.logger.info(f"노출 설정: {self.manual_exposure}")

                return jsonify({'success': True, 'exposure': self.manual_exposure})
            except Exception as e:
                self.logger.error(f"노출 설정 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_gain', methods=['POST'])
        def set_gain():
            """게인 값 설정"""
            try:
                data = request.json
                gain = int(data.get('gain', 50))
                gain = max(0, min(100, gain))

                self.manual_gain = gain
                if not self.auto_exposure:
                    self.camera.set_manual_exposure(self.manual_exposure, self.manual_gain)
                    self.logger.info(f"게인 설정: {self.manual_gain}")

                return jsonify({'success': True, 'gain': self.manual_gain})
            except Exception as e:
                self.logger.error(f"게인 설정 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_brightness', methods=['POST'])
        def set_brightness():
            """밝기 값 설정"""
            try:
                data = request.json
                brightness = int(data.get('brightness', 4))
                brightness = max(0, min(8, brightness))

                self.manual_brightness = brightness
                self.camera.set_brightness(self.manual_brightness)
                self.logger.info(f"밝기 설정: {self.manual_brightness}")

                return jsonify({'success': True, 'brightness': self.manual_brightness})
            except Exception as e:
                self.logger.error(f"밝기 설정 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_contrast', methods=['POST'])
        def set_contrast():
            """대비 값 설정"""
            try:
                data = request.json
                contrast = int(data.get('contrast', 4))
                contrast = max(0, min(8, contrast))

                self.manual_contrast = contrast
                self.camera.set_contrast(self.manual_contrast)
                self.logger.info(f"대비 설정: {self.manual_contrast}")

                return jsonify({'success': True, 'contrast': self.manual_contrast})
            except Exception as e:
                self.logger.error(f"대비 설정 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_sharpness', methods=['POST'])
        def set_sharpness():
            """선명도 값 설정"""
            try:
                data = request.json
                sharpness = int(data.get('sharpness', 4))
                sharpness = max(0, min(8, sharpness))

                self.manual_sharpness = sharpness
                self.camera.set_sharpness(self.manual_sharpness)
                self.logger.info(f"선명도 설정: {self.manual_sharpness}")

                return jsonify({'success': True, 'sharpness': self.manual_sharpness})
            except Exception as e:
                self.logger.error(f"선명도 설정 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/reset', methods=['POST'])
        def reset_values():
            """노출/게인/밝기/대비/선명도 리셋"""
            try:
                self.manual_exposure = 50
                self.manual_gain = 50
                self.manual_brightness = 4
                self.manual_contrast = 4
                self.manual_sharpness = 4

                if not self.auto_exposure:
                    self.camera.set_manual_exposure(self.manual_exposure, self.manual_gain)

                self.camera.set_brightness(self.manual_brightness)
                self.camera.set_contrast(self.manual_contrast)
                self.camera.set_sharpness(self.manual_sharpness)

                self.logger.info("리셋: exposure=50, gain=50, brightness=4, contrast=4, sharpness=4")

                return jsonify({
                    'success': True,
                    'exposure': self.manual_exposure,
                    'gain': self.manual_gain,
                    'brightness': self.manual_brightness,
                    'contrast': self.manual_contrast,
                    'sharpness': self.manual_sharpness
                })
            except Exception as e:
                self.logger.error(f"리셋 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/save_settings', methods=['POST'])
        def save_settings():
            """현재 설정 저장"""
            try:
                self._save_current_settings()
                return jsonify({'success': True})
            except Exception as e:
                self.logger.error(f"설정 저장 실패: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/shutdown', methods=['POST'])
        def shutdown_server():
            """서버 종료"""
            self.shutdown_requested.set()
            return jsonify({'success': True})

    def _generate_stream(self):
        """비디오 스트림 생성"""
        while self.running.is_set():
            with self.state_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    time.sleep(0.01)
                    continue

            # JPEG 인코딩
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

    def _camera_loop(self):
        """
        카메라 프레임 처리 루프 (백그라운드 스레드)

        이 루프는 백그라운드에서 계속 실행되며:
        1. ZED 카메라에서 프레임 캡처
        2. 밝기 통계 계산 (mean, std)
        3. 카메라 설정값 조회 (exposure, gain 등)
        4. ArUco 마커 감지 시도
        5. 통계 업데이트 및 시각화
        6. 웹 UI로 전송할 프레임 저장

        프레임 처리 주기: ~30 FPS
        """
        try:
            self.logger.info("카메라 루프 시작")

            while self.running.is_set():
                try:
                    # ===== 1단계: 프레임 캡처 =====
                    image, depth_map, confidence_map = self.camera.get_frame()
                    self.frame_count += 1

                    # ===== 2단계: 밝기 분석 =====
                    # 그레이스케일로 변환하여 전체 프레임의 밝기 계산
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brightness_mean = float(gray.mean())  # 평균 밝기 (0-255)
                    brightness_std = float(gray.std())    # 표준편차 (밝기 편차)

                    # ===== 3단계: 카메라 설정 조회 =====
                    # ZED SDK에서 현재 적용된 카메라 설정값 가져오기
                    settings = self.camera.get_camera_settings()
                    exposure = settings['exposure']      # 현재 노출값
                    gain = settings['gain']              # 현재 게인값
                    aec_agc = settings['aec_agc']        # Auto Exposure 활성화 여부 (1=Auto, 0=Manual)
                    brightness = settings['brightness']  # 밝기 (0-8)
                    contrast = settings['contrast']      # 대비 (0-8)
                    sharpness = settings['sharpness']    # 선명도 (0-8)

                    # ===== 4단계: ArUco 마커 감지 =====
                    # 이 부분이 핵심! 마커 검출 성공 여부로 노출 설정의 적합성 판단
                    detection = self.detector.detect(image)
                    marker_detected = detection is not None and detection.is_valid

                    # 마커 감지 성공 시 카운트 증가
                    if marker_detected:
                        self.detection_count += 1

                    # ===== 5단계: 통계 누적 및 계산 =====
                    # 최근 100프레임의 이동 평균을 계산하여 안정적인 통계 제공
                    self.brightness_history.append(brightness_mean)
                    self.exposure_history.append(exposure)
                    self.gain_history.append(gain)

                    # 메모리 관리: 최근 100프레임만 유지
                    max_history = 100
                    if len(self.brightness_history) > max_history:
                        self.brightness_history.pop(0)
                        self.exposure_history.pop(0)
                        self.gain_history.pop(0)

                    # 평균값 계산 (최근 100프레임 기준)
                    brightness_avg = float(np.mean(self.brightness_history)) if self.brightness_history else 0
                    exposure_avg = float(np.mean(self.exposure_history)) if self.exposure_history else 0
                    gain_avg = float(np.mean(self.gain_history)) if self.gain_history else 0

                    # 마커 검출률 = (검출 성공 횟수 / 총 프레임) × 100
                    # 이 값이 90% 이상이면 현재 노출 설정이 적합함
                    detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0

                    # 시각화
                    display = self._draw_debug_info(
                        image.copy(),
                        brightness_mean,
                        brightness_std,
                        exposure,
                        gain,
                        aec_agc,
                        marker_detected,
                        detection
                    )

                    # 상태 업데이트
                    with self.state_lock:
                        self.current_frame = display
                        self.current_stats = {
                            'frame_count': self.frame_count,
                            'brightness_mean': brightness_mean,
                            'brightness_std': brightness_std,
                            'brightness_avg': brightness_avg,
                            'exposure': exposure,
                            'gain': gain,
                            'exposure_avg': exposure_avg,
                            'gain_avg': gain_avg,
                            'mode': 'auto' if aec_agc == 1 else 'manual',
                            'marker_detected': marker_detected,
                            'detection_count': self.detection_count,
                            'detection_rate': detection_rate,
                            'manual_exposure': self.manual_exposure,
                            'manual_gain': self.manual_gain,
                            'brightness': brightness,
                            'contrast': contrast,
                            'sharpness': sharpness,
                            'manual_brightness': self.manual_brightness,
                            'manual_contrast': self.manual_contrast,
                            'manual_sharpness': self.manual_sharpness
                        }

                    time.sleep(0.033)  # ~30 FPS

                except ZEDCameraError as e:
                    self.logger.error(f"프레임 캡처 실패: {e}")
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"카메라 루프 오류: {e}", exc_info=True)
        finally:
            self.logger.info("카메라 루프 종료")

    def _draw_debug_info(
        self,
        image: np.ndarray,
        brightness_mean: float,
        brightness_std: float,
        exposure: int,
        gain: int,
        aec_agc: int,
        marker_detected: bool,
        detection
    ) -> np.ndarray:
        """디버깅 정보 오버레이"""
        h, w = image.shape[:2]

        # 배경 바
        cv2.rectangle(image, (0, 0), (w, 200), (40, 40, 40), -1)

        y = 35
        line_height = 35

        # 모드 표시
        mode_text = "AUTO EXPOSURE" if aec_agc == 1 else "MANUAL EXPOSURE"
        mode_color = (0, 255, 255) if aec_agc == 1 else (0, 255, 0)
        cv2.putText(
            image, f"Mode: {mode_text}", (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2
        )
        y += line_height

        # 밝기
        brightness_avg = np.mean(self.brightness_history) if self.brightness_history else 0
        cv2.putText(
            image, f"Brightness: {brightness_mean:.1f} (avg: {brightness_avg:.1f}, std: {brightness_std:.1f})",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        y += line_height

        # 노출/게인
        exposure_avg = np.mean(self.exposure_history) if self.exposure_history else 0
        gain_avg = np.mean(self.gain_history) if self.gain_history else 0
        cv2.putText(
            image, f"Exposure: {exposure} (avg: {exposure_avg:.1f}) | Gain: {gain} (avg: {gain_avg:.1f})",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        y += line_height

        # 마커 검출
        marker_color = (0, 255, 0) if marker_detected else (0, 0, 255)
        marker_text = "Marker: DETECTED" if marker_detected else "Marker: NOT FOUND"
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(
            image, f"{marker_text} (rate: {detection_rate:.1f}%)",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2
        )
        y += line_height

        # 프레임 수
        cv2.putText(
            image, f"Frames: {self.frame_count}",
            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        # 마커 그리기
        if marker_detected and detection:
            self.detector.draw_marker(image, detection, (0, 255, 0))

        return image

    def _save_current_settings(self):
        """
        현재 설정을 권장값으로 저장

        "설정 저장" 버튼을 누르면 호출되며, 현재 카메라 설정과
        마커 검출 통계를 터미널에 출력합니다.

        출력 정보:
            - 현재 모드 (Auto/Manual)
            - 노출/게인 값
            - 밝기/대비/선명도 값
            - 평균 밝기
            - 마커 검출률
            - zed_camera.py에 적용할 코드 예시
        """
        # 현재 카메라 설정 조회
        settings = self.camera.get_exposure_settings()
        brightness_avg = np.mean(self.brightness_history) if self.brightness_history else 0
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0

        # 터미널에 보기 좋게 출력
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📝 현재 설정 저장")
        self.logger.info("=" * 60)
        self.logger.info(f"모드: {'AUTO' if settings['aec_agc'] == 1 else 'MANUAL'}")
        self.logger.info(f"노출 (Exposure): {settings['exposure']}")
        self.logger.info(f"게인 (Gain): {settings['gain']}")
        self.logger.info(f"밝기 (Brightness): {self.manual_brightness}")
        self.logger.info(f"대비 (Contrast): {self.manual_contrast}")
        self.logger.info(f"선명도 (Sharpness): {self.manual_sharpness}")
        self.logger.info(f"평균 밝기: {brightness_avg:.1f}")
        self.logger.info(f"마커 검출률: {detection_rate:.1f}%")
        self.logger.info("=" * 60)
        self.logger.info("")
        self.logger.info("💡 권장 사항:")
        self.logger.info("   app/camera/zed_camera.py의 open() 메서드 line 136에 추가:")
        self.logger.info("")
        self.logger.info(f"   # ===== 최적 카메라 설정 적용 =====")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, {settings['exposure']})")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, {settings['gain']})")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, {self.manual_brightness})")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, {self.manual_contrast})")
        self.logger.info(f"   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, {self.manual_sharpness})")
        self.logger.info(f"   self.logger.info('[카메라 설정] 최적값 적용 완료')")
        self.logger.info("")
        self.logger.info("=" * 60)

    def _run_flask_server(self):
        """Flask 서버 실행"""
        try:
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"Flask 서버 오류: {e}", exc_info=True)

    def run(self):
        """애플리케이션 실행"""
        try:
            # 카메라 초기화
            self.logger.info("=" * 60)
            self.logger.info("노출 디버깅 모드 시작 (웹 버전)")
            self.logger.info("=" * 60)
            self.camera.open()

            # 카메라 루프 시작
            self.running.set()
            camera_thread = Thread(target=self._camera_loop, daemon=True)
            camera_thread.start()

            # Flask 서버 시작
            self.logger.info(f"웹 서버 시작: http://0.0.0.0:{self.port}")
            self.logger.info("브라우저에서 접속하여 노출/게인을 조정하세요")
            self.logger.info("=" * 60)

            flask_thread = Thread(target=self._run_flask_server, daemon=True)
            flask_thread.start()

            # 종료 대기
            while self.running.is_set() and not self.shutdown_requested.is_set():
                time.sleep(0.5)

        except KeyboardInterrupt:
            self.logger.info("\nCtrl+C 감지")
        finally:
            self.shutdown()

    def shutdown(self):
        """애플리케이션 종료"""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("종료 중...")
        self.logger.info("=" * 60)

        self.running.clear()

        # 요약 통계
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count * 100)
            brightness_avg = np.mean(self.brightness_history)
            brightness_std_overall = np.std(self.brightness_history)
            exposure_avg = np.mean(self.exposure_history)
            exposure_std = np.std(self.exposure_history)

            self.logger.info("")
            self.logger.info("📊 세션 요약:")
            self.logger.info(f"  • 총 프레임: {self.frame_count}")
            self.logger.info(f"  • 마커 검출: {self.detection_count}회 ({detection_rate:.1f}%)")
            self.logger.info(f"  • 평균 밝기: {brightness_avg:.1f} ± {brightness_std_overall:.1f}")
            self.logger.info(f"  • 평균 노출: {exposure_avg:.1f} ± {exposure_std:.1f}")
            self.logger.info("=" * 60)

        self.camera.close()
        self.logger.info("✓ 종료 완료")

        time.sleep(0.5)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='ZED 카메라 노출 디버깅 도구 (웹 버전)')
    parser.add_argument(
        '-c', '--config',
        default='config/config.yaml',
        help='설정 파일 경로 (기본값: config/config.yaml)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5001,
        help='웹 서버 포트 (기본값: 5001)'
    )
    args = parser.parse_args()

    debugger = ExposureDebuggerWeb(config_path=args.config, port=args.port)
    debugger.run()


if __name__ == '__main__':
    main()
