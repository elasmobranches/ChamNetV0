"""
Flask Web Streaming Server
==========================
웹 브라우저로 카메라 스트림을 보고 원격 제어하는 Flask 서버

기능:
1. MJPEG 스트리밍: 실시간 카메라 영상 웹 브라우저로 전송
2. REST API: 거리 설정, 프레임 저장, 모드 변경, 종료 등
3. 상태 조회: 마커 감지 여부, ZED depth, 오차율, 저장 횟수

엔드포인트:
- GET  /              : 메인 웹 페이지
- GET  /video_feed    : MJPEG 스트리밍
- GET  /api/status    : 현재 상태 조회
- POST /api/set_distance : 거리 설정
- POST /api/save_frame   : 프레임 저장
- POST /api/toggle_mode  : 측정 모드 토글
- POST /api/shutdown     : 서버 종료

웹 UI 키보드 단축키:
- D: 거리 입력란 포커스
- S: 프레임 저장
- M: 모드 변경
- Q: 종료
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
    """
    Flask 웹 스트리밍 서버 클래스

    메인 애플리케이션(DepthEstimationApp)에서 이 클래스를 사용하여
    웹 인터페이스 제공.

    사용 패턴:
        server = StreamServer(config)
        server.on_distance_input = callback_func  # 콜백 등록
        server.update_frame(frame)                # 프레임 업데이트
        server.app.run(...)                       # Flask 실행

    콜백 함수:
        - on_distance_input: 거리 입력 시 호출 (distance: float)
        - on_save_frame: 프레임 저장 시 호출 ()
        - on_toggle_mode: 모드 변경 시 호출 ()

    Attributes:
        app: Flask 애플리케이션 인스턴스
        current_frame: 현재 스트리밍 중인 프레임
        status: 현재 상태 딕셔너리
    """

    def __init__(self, config: Config):
        """
        스트림 서버 초기화

        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.web')

        # ========== Flask 앱 설정 ==========
        # template_folder, static_folder는 상대 경로 (이 파일 기준)
        self.app = Flask(
            __name__,
            template_folder='../../templates',  # HTML 템플릿
            static_folder='../../static'        # CSS, JS
        )
        CORS(self.app)  # Cross-Origin Resource Sharing 허용

        # ========== 상태 관리 ==========
        self.current_frame: Optional[np.ndarray] = None  # 스트리밍 프레임
        self.frame_lock = Lock()  # 프레임 접근 동기화
        self.shutdown_event = Event()  # 종료 신호

        # ========== 콜백 함수 (메인 앱에서 등록) ==========
        self.on_distance_input: Optional[Callable[[float], None]] = None
        self.on_save_frame: Optional[Callable[[], None]] = None
        self.on_toggle_mode: Optional[Callable[[], None]] = None
        self.on_set_marker_size: Optional[Callable[[int], None]] = None

        # ========== 상태 정보 (API로 조회 가능) ==========
        self.status = {
            'current_distance': None,  # 설정된 거리 (m)
            'marker_size_mm': config.marker_size_mm,  # 현재 마커 크기 (mm)
            'saved_count': 0,          # 저장된 프레임 수
            'mode': 'marker_region',   # 측정 모드
            'marker_detected': False,  # 마커 감지 여부
            'zed_depth': None,         # ZED depth 값 (mm)
            'error_pct': None,         # 오차율 (%)
            'marker_center_x': None,   # 마커 중심 X 좌표 (픽셀)
            'marker_center_y': None,   # 마커 중심 Y 좌표 (픽셀)
            'marker_angle': None       # 마커 회전 각도 (도)
        }
        self.status_lock = Lock()  # 상태 접근 동기화

        # ========== Flask 라우트 등록 ==========
        self._register_routes()

        self.logger.info("Flask 웹 서버 초기화 완료")

    def _register_routes(self):
        """
        Flask 라우트(URL 핸들러) 등록

        모든 API 엔드포인트를 정의.
        """

        @self.app.route('/')
        def index():
            """
            메인 페이지 (웹 UI)

            Returns:
                HTML: index.html 렌더링
            """
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            """
            MJPEG 비디오 스트리밍 엔드포인트

            웹 브라우저의 <img src="/video_feed">에서 호출됨.
            무한 루프로 JPEG 프레임을 연속 전송.

            Returns:
                Response: multipart/x-mixed-replace MIME 타입 스트림
            """
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/api/status')
        def get_status():
            """
            현재 상태 조회 API

            웹 UI에서 주기적으로 폴링하여 상태 표시.

            Returns:
                JSON: 상태 딕셔너리
            """
            with self.status_lock:
                return jsonify(self.status)

        @self.app.route('/api/set_distance', methods=['POST'])
        def set_distance():
            """
            거리 설정 API

            Request Body:
                {"distance": 1.5}  // 미터 단위

            Returns:
                JSON: {"success": true, "distance": 1.5}
            """
            try:
                data = request.get_json()
                distance = float(data.get('distance', 0))

                # 유효성 검사
                if distance <= 0:
                    return jsonify({'success': False, 'error': '거리는 0보다 커야 합니다'}), 400

                # 콜백 호출 (메인 앱의 analyzer.set_distance)
                if self.on_distance_input:
                    self.on_distance_input(distance)

                # 상태 업데이트
                with self.status_lock:
                    self.status['current_distance'] = distance

                self.logger.info(f"거리 설정: {distance}m")
                return jsonify({'success': True, 'distance': distance})

            except Exception as e:
                self.logger.error(f"거리 설정 오류: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/save_frame', methods=['POST'])
        def save_frame():
            """
            프레임 저장 API

            현재 프레임을 이미지로 저장하고 측정 기록 추가.
            여러 조건 검증 후 저장 수행.

            Returns:
                JSON: {"success": true, "saved_count": 5}

            Error Codes:
                400: 사용자 입력 오류 (거리 미설정, 마커 미감지 등)
                500: 시스템 오류
            """
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
            """
            측정 모드 토글 API

            마커 영역 모드 ↔ 윈도우 모드 전환.

            Returns:
                JSON: {"success": true, "mode": "window"}
            """
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

        @self.app.route('/api/set_marker_size/<int:size>', methods=['POST'])
        def set_marker_size(size: int):
            """
            마커 크기 설정 API

            Args:
                size: 마커 크기 (mm) - URL 경로로 전달

            Returns:
                JSON: {"success": true, "marker_size_mm": 100}
            """
            try:
                # 허용된 마커 크기 검증
                allowed_sizes = [50, 100, 250]
                if size not in allowed_sizes:
                    return jsonify({
                        'success': False,
                        'error': f'허용되지 않은 마커 크기: {size}mm (허용: {allowed_sizes})'
                    }), 400

                # 콜백 호출 (메인 앱의 config.marker_size_mm 설정)
                if self.on_set_marker_size:
                    self.on_set_marker_size(size)

                # 상태 업데이트
                with self.status_lock:
                    self.status['marker_size_mm'] = size

                self.logger.info(f"마커 크기 설정: {size}mm")
                return jsonify({'success': True, 'marker_size_mm': size})

            except Exception as e:
                self.logger.error(f"마커 크기 설정 오류: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/download_csv', methods=['POST'])
        def download_csv():
            """
            CSV 다운로드 요청 API

            실제 다운로드는 메인 애플리케이션에서 처리.

            Returns:
                JSON: {"success": true, "message": "..."}
            """
            return jsonify({'success': True, 'message': 'CSV 저장이 요청되었습니다'})

        @self.app.route('/api/shutdown', methods=['POST'])
        def shutdown():
            """
            서버 종료 API

            종료 이벤트를 설정하여 메인 스레드에 알림.
            CSV 자동 저장은 메인 앱의 shutdown()에서 처리.

            Returns:
                JSON: {"success": true, "message": "..."}
            """
            self.logger.info("=" * 60)
            self.logger.info("웹 UI에서 종료 요청 수신")
            self.logger.info("=" * 60)
            self.shutdown_event.set()

            # werkzeug 개발 서버 종료 (있으면)
            func = request.environ.get('werkzeug.server.shutdown')
            if func is not None:
                func()

            return jsonify({'success': True, 'message': 'CSV 파일이 자동으로 저장되고 서버가 종료됩니다'})

    def _generate_frames(self):
        """
        MJPEG 프레임 제너레이터

        무한 루프로 현재 프레임을 JPEG로 인코딩하여 yield.
        프레임이 없으면 "Waiting for camera..." 메시지 표시.

        Yields:
            bytes: MJPEG 프레임 바이트
        """
        while not self.shutdown_event.is_set():
            try:
                # 현재 프레임 가져오기 (thread-safe)
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        # 기본 프레임 (카메라 대기 중)
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

                # MJPEG 프레임 형식으로 yield
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                self.logger.error(f"프레임 생성 오류: {e}")
                self.logger.debug(traceback.format_exc())

    def update_frame(self, frame: np.ndarray) -> None:
        """
        스트리밍 프레임 업데이트

        카메라 루프에서 호출하여 새 프레임 설정.

        Args:
            frame: BGR 이미지 (numpy array)
        """
        with self.frame_lock:
            self.current_frame = frame.copy()

    def update_status(
        self,
        marker_detected: bool = None,
        zed_depth: float = None,
        error_pct: float = None,
        saved_count: int = None,
        marker_center_x: int = None,
        marker_center_y: int = None,
        marker_angle: float = None
    ) -> None:
        """
        상태 정보 업데이트

        카메라 루프에서 호출하여 실시간 상태 갱신.

        Args:
            marker_detected: 마커 감지 여부
            zed_depth: ZED depth 값 (mm)
            error_pct: 오차율 (%)
            saved_count: 저장된 프레임 수
            marker_center_x: 마커 중심 X 좌표 (픽셀)
            marker_center_y: 마커 중심 Y 좌표 (픽셀)
            marker_angle: 마커 회전 각도 (도)
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
            if marker_center_x is not None:
                self.status['marker_center_x'] = marker_center_x
            if marker_center_y is not None:
                self.status['marker_center_y'] = marker_center_y
            if marker_angle is not None:
                self.status['marker_angle'] = marker_angle

    def get_current_distance(self) -> Optional[float]:
        """
        현재 설정된 거리 조회

        Returns:
            float: 설정된 거리 (m), 없으면 None
        """
        with self.status_lock:
            return self.status['current_distance']

    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: bool = False):
        """
        Flask 서버 실행

        보통은 메인 앱에서 별도 스레드로 app.run()을 직접 호출함.
        이 메서드는 독립 실행 시 사용.

        Args:
            host: 호스트 주소 (None이면 설정값 사용)
            port: 포트 번호 (None이면 설정값 사용)
            debug: 디버그 모드
        """
        if host is None:
            host = self.config.web_host
        if port is None:
            port = self.config.web_port

        self.logger.info(f"Flask 서버 시작: http://{host}:{port}")

        # Flask 기본 로거 비활성화
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True,       # 멀티스레드 요청 처리
                use_reloader=False   # 자동 리로드 비활성화
            )
        except Exception as e:
            self.logger.error(f"Flask 서버 오류: {e}")
            raise

    def is_shutdown_requested(self) -> bool:
        """
        종료가 요청되었는지 확인

        메인 스레드에서 폴링하여 종료 조건 체크.

        Returns:
            bool: 종료 요청됨 True
        """
        return self.shutdown_event.is_set()
