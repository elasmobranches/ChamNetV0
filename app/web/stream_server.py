"""
Flask web streaming server
WiFi 없는 환경에서 웹 브라우저를 통해 카메라 스트림을 보고 키보드로 조작할 수 있습니다.
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from threading import Lock, Event
from typing import Optional, Callable
import traceback
from ..utils.logger import get_logger
from ..utils.config import Config


class StreamServer:
    """Flask 웹 스트리밍 서버"""

    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.web')

        # Flask 앱 설정
        self.app = Flask(
            __name__,
            template_folder='../../templates',
            static_folder='../../static'
        )
        CORS(self.app)

        # 상태 관리
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = Lock()
        self.shutdown_event = Event()

        # 콜백 함수들
        self.on_distance_input: Optional[Callable[[float], None]] = None
        self.on_save_frame: Optional[Callable[[], None]] = None
        self.on_toggle_mode: Optional[Callable[[], None]] = None

        # 상태 정보
        self.status = {
            'current_distance': None,
            'saved_count': 0,
            'mode': 'marker_region',
            'marker_detected': False,
            'zed_depth': None,
            'error_pct': None
        }
        self.status_lock = Lock()

        # 라우트 등록
        self._register_routes()

        self.logger.info("Flask 웹 서버 초기화 완료")

    def _register_routes(self):
        """Flask 라우트를 등록합니다."""

        @self.app.route('/')
        def index():
            """메인 페이지"""
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            """비디오 스트리밍 엔드포인트"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/api/status')
        def get_status():
            """현재 상태 반환"""
            with self.status_lock:
                return jsonify(self.status)

        @self.app.route('/api/set_distance', methods=['POST'])
        def set_distance():
            """거리 설정"""
            try:
                data = request.get_json()
                distance = float(data.get('distance', 0))

                if distance <= 0:
                    return jsonify({'success': False, 'error': '거리는 0보다 커야 합니다'}), 400

                # 콜백 호출
                if self.on_distance_input:
                    self.on_distance_input(distance)

                with self.status_lock:
                    self.status['current_distance'] = distance

                self.logger.info(f"거리 설정: {distance}m")
                return jsonify({'success': True, 'distance': distance})

            except Exception as e:
                self.logger.error(f"거리 설정 오류: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/save_frame', methods=['POST'])
        def save_frame():
            """현재 프레임 저장"""
            try:
                if not self.on_save_frame:
                    raise RuntimeError("프레임 저장 콜백이 등록되지 않았습니다.")

                # 콜백 실행 (예외 발생 시 catch됨)
                self.on_save_frame()

                # 성공 시에만 카운트 증가
                with self.status_lock:
                    self.status['saved_count'] += 1
                    count = self.status['saved_count']

                self.logger.info(f"✓ 프레임 저장 성공 (#{count})")

                # 명시적으로 200 상태 코드 반환
                response = jsonify({'success': True, 'saved_count': count})
                response.status_code = 200
                return response

            except ValueError as e:
                # 사용자 입력 오류 (거리 미설정, 마커 미감지 등)
                error_msg = str(e)
                self.logger.warning(f"⚠️  프레임 저장 실패: {error_msg}")

                response = jsonify({'success': False, 'error': error_msg})
                response.status_code = 400
                return response

            except Exception as e:
                # 기타 시스템 오류
                error_msg = str(e)
                self.logger.error(f"❌ 프레임 저장 오류: {error_msg}", exc_info=True)

                response = jsonify({'success': False, 'error': f"시스템 오류: {error_msg}"})
                response.status_code = 500
                return response

        @self.app.route('/api/toggle_mode', methods=['POST'])
        def toggle_mode():
            """측정 모드 토글"""
            try:
                if self.on_toggle_mode:
                    self.on_toggle_mode()

                with self.status_lock:
                    current_mode = self.status['mode']
                    new_mode = 'window' if current_mode == 'marker_region' else 'marker_region'
                    self.status['mode'] = new_mode

                self.logger.info(f"모드 변경: {new_mode}")
                return jsonify({'success': True, 'mode': new_mode})

            except Exception as e:
                self.logger.error(f"모드 변경 오류: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/download_csv', methods=['POST'])
        def download_csv():
            """CSV 다운로드 요청"""
            # 이 기능은 메인 애플리케이션에서 처리
            return jsonify({'success': True, 'message': 'CSV 저장이 요청되었습니다'})

        @self.app.route('/api/shutdown', methods=['POST'])
        def shutdown():
            """서버 종료"""
            self.logger.info("=" * 60)
            self.logger.info("웹 UI에서 종료 요청 수신")
            self.logger.info("=" * 60)
            self.shutdown_event.set()

            # Flask 서버 종료 함수
            func = request.environ.get('werkzeug.server.shutdown')
            if func is not None:
                func()

            return jsonify({'success': True, 'message': 'CSV 파일이 자동으로 저장되고 서버가 종료됩니다'})

    def _generate_frames(self):
        """프레임 생성 제너레이터"""
        while not self.shutdown_event.is_set():
            try:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        # 기본 프레임 (검은 화면)
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            frame,
                            'Waiting for camera...',
                            (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )

                # JPEG 인코딩
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.stream_quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)

                # 프레임 전송
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                self.logger.error(f"프레임 생성 오류: {e}")
                self.logger.debug(traceback.format_exc())

    def update_frame(self, frame: np.ndarray) -> None:
        """
        현재 프레임을 업데이트합니다.

        Args:
            frame: 새 프레임 (BGR 이미지)
        """
        with self.frame_lock:
            self.current_frame = frame.copy()

    def update_status(
        self,
        marker_detected: bool = None,
        zed_depth: float = None,
        error_pct: float = None,
        saved_count: int = None
    ) -> None:
        """
        상태 정보를 업데이트합니다.

        Args:
            marker_detected: 마커 감지 여부
            zed_depth: ZED depth 값 (mm)
            error_pct: 오차율 (%)
            saved_count: 저장된 프레임 수
        """
        with self.status_lock:
            if marker_detected is not None:
                self.status['marker_detected'] = marker_detected
            if zed_depth is not None:
                self.status['zed_depth'] = zed_depth
            if error_pct is not None:
                self.status['error_pct'] = error_pct
            if saved_count is not None:
                self.status['saved_count'] = saved_count

    def get_current_distance(self) -> Optional[float]:
        """현재 설정된 거리를 반환합니다."""
        with self.status_lock:
            return self.status['current_distance']

    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: bool = False):
        """
        Flask 서버를 실행합니다.

        Args:
            host: 호스트 주소
            port: 포트 번호
            debug: 디버그 모드
        """
        if host is None:
            host = self.config.web_host
        if port is None:
            port = self.config.web_port

        self.logger.info(f"Flask 서버 시작: http://{host}:{port}")

        # Flask 로거 비활성화 (우리의 로거 사용)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True,
                use_reloader=False  # 리로더 비활성화
            )
        except Exception as e:
            self.logger.error(f"Flask 서버 오류: {e}")
            raise

    def is_shutdown_requested(self) -> bool:
        """종료가 요청되었는지 확인합니다."""
        return self.shutdown_event.is_set()
