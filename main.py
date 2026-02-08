#!/usr/bin/env python3
"""
ZED ArUco Depth Estimation - Main Application
==============================================
ZED 2i 카메라와 ArUco 마커를 이용한 거리 측정 시스템

주요 기능:
- ZED 카메라로 RGB 이미지와 Depth 맵 동시 캡처
- ArUco 마커 감지 및 위치 추적
- 마커 영역의 Depth 값 통계 분석 (중앙값, 평균, 표준편차)
- 실제 거리와 ZED Depth 비교를 통한 오차율 계산
- Flask 웹 서버로 실시간 스트리밍 및 원격 제어
- 측정 결과 CSV 자동 저장

사용법:
    python3 main.py                    # 기본 실행
    python3 main.py -p 8080            # 포트 변경
    python3 main.py -c my_config.yaml  # 설정 파일 지정
"""

import cv2
import numpy as np
import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime
from threading import Thread, Event, Lock
import time

from app.utils.config import Config, ConfigError
from app.utils.logger import setup_logger, get_logger
from app.camera import ZEDCamera, ZEDCameraError
from app.marker import ArucoDetector, MarkerDetection
from app.measurement import DepthAnalyzer, DepthStats
from app.web import StreamServer


class DepthEstimationApp:
    """
    메인 애플리케이션 클래스

    전체 시스템을 통합하고 관리하는 핵심 클래스.
    카메라, 마커 감지, Depth 분석, 웹 서버를 조율한다.

    구조:
        ┌─────────────────────────────────────────────────────────┐
        │                    DepthEstimationApp                    │
        │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
        │  │ ZEDCamera│  │ArucoDetector│ │DepthAnalyzer│ │StreamServer│ │
        │  └────┬────┘  └─────┬─────┘  └─────┬─────┘  └────┬─────┘  │
        │       │              │              │              │       │
        │       └──────────────┴──────────────┴──────────────┘       │
        │                           │                                 │
        │                    _process_frame()                        │
        └─────────────────────────────────────────────────────────────┘

    스레드 구조:
        - 메인 스레드: 종료 신호 대기
        - 카메라 스레드: 프레임 캡처 및 처리 (daemon)
        - Flask 스레드: 웹 서버 실행 (daemon)
    """

    def __init__(self, config_path: str = None):
        """
        애플리케이션 초기화

        Args:
            config_path: 설정 파일 경로 (None이면 config/config.yaml 사용)
        """
        # ========== 1단계: 설정 로드 ==========
        try:
            self.config = Config(config_path)
        except ConfigError as e:
            print(f"❌ 설정 로드 실패: {e}")
            sys.exit(1)

        # ========== 2단계: 로거 초기화 ==========
        # 콘솔 + 파일 동시 출력, 일별 로그 파일 생성
        self.logger = setup_logger(
            name='depth_estimation',
            log_dir=self.config.log_dir,
            level='INFO'
        )

        self.logger.info("=" * 60)
        self.logger.info("ZED ArUco Depth Estimation v2.0 시작")
        self.logger.info("=" * 60)

        # ========== 3단계: 세션 디렉토리 생성 ==========
        # 실행 시마다 타임스탬프 폴더 생성 (예: data/results/20240115_143025/)
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.config.output_dir / self.session_timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"세션 디렉토리: {self.session_dir}")

        # ========== 4단계: 핵심 컴포넌트 초기화 ==========
        self.camera = ZEDCamera(self.config)      # ZED 카메라 제어
        self.detector = ArucoDetector(self.config) # ArUco 마커 감지
        self.analyzer = DepthAnalyzer(self.config) # Depth 통계 분석
        self.server = StreamServer(self.config)    # Flask 웹 서버

        # ========== 4단계: 웹 서버 콜백 등록 ==========
        # 웹 UI에서 버튼 클릭 시 호출될 함수들
        self.server.on_distance_input = self._on_distance_input    # 거리 입력
        self.server.on_save_frame = self._on_save_frame            # 프레임 저장
        self.server.on_toggle_mode = self._on_toggle_mode          # 모드 변경
        self.server.on_set_marker_size = self._on_set_marker_size  # 마커 크기 변경

        # ========== 5단계: 상태 변수 초기화 ==========
        self.running = Event()          # 실행 중 플래그 (스레드 간 동기화)
        self.state_lock = Lock()        # 공유 상태 보호용 락
        self.current_frame_display = None   # 현재 화면에 표시 중인 프레임
        self.current_detection = None       # 현재 마커 감지 결과
        self.current_depth_stats = None     # 현재 Depth 통계
        self.shutdown_called = False    # shutdown 중복 호출 방지 플래그
        self.camera_thread = None       # 카메라 루프 스레드 참조

        # ========== 6단계: 종료 시그널 핸들러 등록 ==========
        # Ctrl+C (SIGINT) 또는 kill (SIGTERM) 시 graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        시그널 핸들러 (Ctrl+C, kill 등)

        프로세스 종료 신호를 받으면 shutdown() 호출하여
        카메라 정리, CSV 저장 등 cleanup 수행
        """
        self.logger.info("종료 시그널 수신")
        self.shutdown()

    def _on_distance_input(self, distance: float):
        """
        거리 입력 콜백 (웹 UI에서 거리 설정 시 호출)

        레이저 측정기로 측정한 실제 거리를 입력받아
        ZED Depth와 비교할 기준값으로 사용

        Args:
            distance: 실제 거리 (미터 단위)
        """
        self.analyzer.set_distance(distance)

    def _on_save_frame(self):
        """
        프레임 저장 콜백 (웹 UI에서 'S' 키 또는 저장 버튼 클릭 시)

        현재 프레임을 이미지로 저장하고, 측정 데이터를 기록한다.
        여러 조건을 검증하여 유효한 상태에서만 저장 수행.

        Raises:
            ValueError: 마커 미감지, Depth 데이터 없음, 거리 미설정 등
            IOError: 이미지 파일 저장 실패
            RuntimeError: 측정 기록 추가 실패
        """
        # ===== Lock으로 보호된 상태 읽기 (Race Condition 방지) =====
        # 카메라 스레드가 업데이트 중인 상태를 읽으면 inconsistent할 수 있음
        with self.state_lock:
            # ----- 저장 조건 검증 -----
            if not self.current_detection:
                raise ValueError("마커가 감지되지 않았습니다. 마커를 카메라에 보이도록 배치하세요.")

            if not self.current_depth_stats:
                raise ValueError("Depth 데이터가 없습니다.")

            if self.analyzer.current_distance_m is None:
                raise ValueError("거리가 설정되지 않았습니다. 먼저 거리를 입력하세요 (D 키).")

            if self.current_frame_display is None:
                raise ValueError("프레임 데이터가 없습니다.")

            # ----- 복사본 생성 (lock 해제 후 안전하게 사용) -----
            frame_display = self.current_frame_display.copy()
            depth_stats = self.current_depth_stats
            detection = self.current_detection

        # ===== 이미지 파일 저장 =====
        # 파일명 형식: {거리}m_{시간}.jpg (예: 1.500m_143025.jpg)
        # 저장 위치: 세션 디렉토리/images/
        distance = self.analyzer.current_distance_m
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{distance:.3f}m_{timestamp}.jpg"

        output_dir = self.session_dir / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
        filepath = output_dir / filename

        success = cv2.imwrite(str(filepath), frame_display)
        if not success:
            raise IOError(f"이미지 저장 실패: {filepath}")

        # ===== 측정 기록 추가 =====
        # CSV에 저장될 데이터: 실제거리, 마커크기, ZED depth, 오차율, 마커 각도 등
        success = self.analyzer.add_measurement(
            depth_stats,
            detection,
            marker_size_mm=self.config.marker_size_mm,  # 현재 설정된 마커 크기
            image_filename=filename  # 이미지 파일명도 기록
        )

        if not success:
            raise RuntimeError("측정 기록 추가에 실패했습니다.")

        self.logger.info(f"✓ 프레임 저장 성공: {filepath}")

    def _on_toggle_mode(self):
        """
        측정 모드 토글 콜백 (웹 UI에서 'M' 키 클릭 시)

        Depth 측정 방식을 전환:
        - 마커 영역 모드: 마커 내부 전체 영역의 Depth 통계 (더 정확)
        - 윈도우 모드: 마커 중심점 주변 작은 윈도우만 사용 (더 빠름)
        """
        current = self.config.use_marker_region
        new_mode = not current
        self.config.use_marker_region = new_mode  # property setter 사용
        mode_text = '마커 영역' if new_mode else '윈도우'
        self.logger.info(f"측정 모드 변경: {mode_text}")

    def _on_set_marker_size(self, size_mm: int):
        """
        마커 크기 설정 콜백 (웹 UI에서 마커 크기 버튼 클릭 시)

        실험 중 다른 크기의 마커로 교체할 때 호출.
        CSV 기록에 정확한 마커 크기가 저장되도록 함.

        Args:
            size_mm: 마커 크기 (mm) - 50, 100, 250 등
        """
        self.config.marker_size_mm = size_mm  # property setter 사용
        self.logger.info(f"마커 크기 변경: {size_mm}mm")

    def _process_frame(self, image: np.ndarray, depth_map: np.ndarray, confidence_map: np.ndarray):
        """
        단일 프레임 처리 (카메라 루프에서 매 프레임마다 호출)

        처리 흐름:
        1. 마커 감지 (ArUco)
        2. 마커 영역 Depth 분석
        3. 오차율 계산 (실제 거리와 비교)
        4. 화면에 시각화 그리기
        5. 웹 서버로 프레임 전송

        Args:
            image: RGB 이미지 (numpy array, shape: H x W x 3)
            depth_map: Depth 맵 (numpy array, shape: H x W, 단위: mm)
            confidence_map: Confidence 맵 (numpy array, shape: H x W, 범위: 0-100)
        """
        display = image.copy()  # 원본 이미지 보존을 위해 복사본에 그림

        # ========== 1단계: 마커 감지 ==========
        detection = self.detector.detect(image)

        marker_detected = False
        depth_stats = None
        error_pct = np.nan  # 오차율 (계산 전에는 NaN)

        # ========== 2단계: 마커가 감지된 경우 처리 ==========
        if detection and detection.is_valid:
            marker_detected = True

            # ----- Depth 분석 (마커 영역 or 윈도우) -----
            depth_stats = self.analyzer.analyze(depth_map, detection, confidence_map)

            if depth_stats:
                # ----- 오차율 계산 -----
                # 오차율 = |ZED측정값 - 실제값| / 실제값 × 100
                if self.analyzer.current_distance_m:
                    actual_mm = self.analyzer.current_distance_m * 1000  # m → mm
                    error_mm = abs(depth_stats.median_mm - actual_mm)
                    error_pct = (error_mm / actual_mm) * 100

                # ----- 오차율에 따른 색상 결정 -----
                # < 2%: 녹색, 2-5%: 노랑, 5-10%: 주황, > 10%: 빨강
                color = self.analyzer.get_error_color(error_pct)

                # ----- 마커 시각화 그리기 -----
                self._draw_marker_visualization(
                    display, detection, depth_stats, error_pct, color
                )

        # ========== 3단계: 상태 바 그리기 ==========
        self._draw_status_bar(display, marker_detected)

        # ========== 4단계: 공유 상태 업데이트 (thread-safe) ==========
        # 세 변수를 atomic하게 한 번에 업데이트해야 일관성 유지
        with self.state_lock:
            self.current_frame_display = display.copy()
            self.current_detection = detection
            self.current_depth_stats = depth_stats

        # ========== 5단계: 웹 서버 업데이트 ==========
        self.server.update_frame(display)  # JPEG 스트리밍용 프레임

        # 상태 정보 (웹 UI에서 API로 조회)
        # numpy 타입을 Python 기본 타입으로 변환 (JSON 직렬화 가능하도록)
        self.server.update_status(
            marker_detected=marker_detected,
            zed_depth=float(depth_stats.median_mm) if depth_stats else None,
            error_pct=float(error_pct) if not np.isnan(error_pct) else None,
            saved_count=len(self.analyzer.records),
            marker_center_x=int(detection.center_x) if detection else None,
            marker_center_y=int(detection.center_y) if detection else None,
            marker_angle=float(detection.angle) if detection else None
        )

    def _draw_marker_visualization(
        self,
        image: np.ndarray,
        detection: MarkerDetection,
        stats: DepthStats,
        error_pct: float,
        color: tuple
    ):
        """
        마커 시각화 그리기

        마커 영역에 반투명 색상 오버레이, 외곽선, 중심점,
        그리고 Depth 값과 오차율 텍스트를 표시

        Args:
            image: 그릴 이미지 (수정됨)
            detection: 마커 감지 결과
            stats: Depth 통계
            error_pct: 오차율 (%)
            color: 표시 색상 (BGR)
        """
        # 마커 영역 그리기 (반투명 + 외곽선 + 중심점)
        self.detector.draw_marker(image, detection, color)

        # ===== 화면 중앙 좌표 계산 (해상도 기반) =====
        width, height = self.config.get_resolution_dimensions()
        screen_center_x = width // 2
        screen_center_y = height // 2

        # 마커 중심에서 화면 중앙까지의 거리 (픽셀)
        offset_x = detection.center_x - screen_center_x
        offset_y = detection.center_y - screen_center_y
        offset_distance = np.sqrt(offset_x**2 + offset_y**2)

        # ===== 정보 텍스트 준비 =====
        cx, cy = detection.center_x, detection.center_y
        zed_m = stats.median_mm / 1000  # mm → m 변환

        text_lines = [
            f"ZED: {zed_m:.3f}m",  # ZED 측정값
            f"Error: {error_pct:.2f}%" if not np.isnan(error_pct) else f"ZED: {zed_m:.3f}m",
            f"Pos: ({detection.center_x}, {detection.center_y}) | Offset: {offset_distance:.0f}px",  # 위치와 중앙으로부터 거리
            f"Angle: {detection.angle:.1f}deg | Std: {stats.std_mm:.1f}mm"  # 각도, 표준편차
        ]

        # ===== 텍스트 그리기 (그림자 효과로 가독성 향상) =====
        for i, text in enumerate(text_lines):
            y_pos = cy + 60 + i * 30  # 마커 아래에 순차적으로 배치

            # 검은색 그림자 (두꺼운 선)
            cv2.putText(
                image, text, (cx + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5
            )
            # 컬러 텍스트 (얇은 선)
            cv2.putText(
                image, text, (cx + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    def _draw_status_bar(self, image: np.ndarray, marker_detected: bool):
        """
        화면 상단 상태 바 그리기

        표시 정보:
        - Target 거리 (설정된 실제 거리)
        - 측정 모드 (마커 영역 / 윈도우)
        - 저장된 프레임 수
        - 마커 감지 상태

        Args:
            image: 그릴 이미지 (수정됨)
            marker_detected: 마커 감지 여부
        """
        h, w = image.shape[:2]

        # 배경 바 (어두운 회색)
        cv2.rectangle(image, (0, 0), (w, 90), (40, 40, 40), -1)

        # ===== 거리 정보 (좌상단) =====
        if self.analyzer.current_distance_m:
            dist_text = f"Target: {self.analyzer.current_distance_m:.3f} m"
            color = (0, 255, 0)  # 녹색 (설정됨)
        else:
            dist_text = "Target: Not Set"
            color = (0, 255, 255)  # 노랑 (미설정)

        cv2.putText(image, dist_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # ===== 통계 정보 (두 번째 줄) =====
        mode_text = "Mode: Marker Region" if self.config.use_marker_region else "Mode: Window"
        marker_size_text = f"Size: {self.config.marker_size_mm}mm"
        saved_text = f"Saved: {len(self.analyzer.records)}"
        marker_text = "Marker: Detected" if marker_detected else "Marker: Not Found"

        cv2.putText(image, mode_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, marker_size_text, (250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(image, saved_text, (420, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(
            image, marker_text, (600, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0) if marker_detected else (0, 0, 255), 1  # 감지: 녹색, 미감지: 빨강
        )

    def _camera_loop(self):
        """
        카메라 루프 (별도 스레드에서 실행)

        무한 루프로 프레임을 캡처하고 처리.
        running Event가 clear되면 종료.
        """
        try:
            self.logger.info("카메라 루프 시작")

            while self.running.is_set():
                try:
                    # 프레임 캡처 (RGB + Depth + Confidence)
                    image, depth_map, confidence_map = self.camera.get_frame()

                    # 프레임 처리 (마커 감지, Depth 분석, 시각화)
                    self._process_frame(image, depth_map, confidence_map)

                    # FPS 제어 (설정된 stream_fps에 맞춤)
                    time.sleep(1.0 / self.config.stream_fps)

                except ZEDCameraError as e:
                    self.logger.error(f"프레임 캡처 실패: {e}")
                    time.sleep(0.1)  # 에러 시 잠시 대기 후 재시도

        except Exception as e:
            self.logger.error(f"카메라 루프 오류: {e}", exc_info=True)
        finally:
            self.logger.info("카메라 루프 종료")

    def _run_flask_server(self):
        """
        Flask 서버 실행 (별도 스레드에서 실행)

        웹 브라우저로 접속하면 실시간 스트리밍 및 제어 UI 제공
        """
        try:
            # Flask 기본 로거 비활성화 (우리 로거만 사용)
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            # Flask 서버 실행
            self.server.app.run(
                host=self.config.web_host,
                port=self.config.web_port,
                debug=False,
                threaded=True,       # 멀티스레드 요청 처리
                use_reloader=False   # 자동 리로드 비활성화 (중복 프로세스 방지)
            )
        except Exception as e:
            self.logger.error(f"Flask 서버 오류: {e}", exc_info=True)

    def run(self):
        """
        애플리케이션 실행 (메인 진입점)

        실행 순서:
        1. ZED 카메라 초기화
        2. 카메라 루프 스레드 시작
        3. Flask 서버 스레드 시작
        4. 종료 신호 대기 (Ctrl+C, 웹 UI 종료 버튼)
        5. shutdown() 호출하여 정리
        """
        try:
            # ========== 카메라 초기화 ==========
            self.logger.info("ZED 카메라 초기화 중...")
            self.camera.open()

            # ========== 카메라 루프 시작 (별도 스레드) ==========
            self.running.set()  # 실행 플래그 ON
            self.camera_thread = Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()

            # ========== Flask 서버 시작 (별도 스레드) ==========
            self.logger.info("Flask 웹 서버 시작...")
            self.logger.info(f"브라우저에서 http://{self.config.web_host}:{self.config.web_port} 접속")
            self.logger.info("종료하려면 Ctrl+C를 누르거나 웹 UI에서 '종료' 버튼을 클릭하세요")

            flask_thread = Thread(target=self._run_flask_server, daemon=True)
            flask_thread.start()

            # ========== 메인 스레드: 종료 신호 대기 ==========
            try:
                while self.running.is_set() and not self.server.is_shutdown_requested():
                    time.sleep(0.5)  # 0.5초마다 종료 조건 체크
            except KeyboardInterrupt:
                self.logger.info("\n")
                self.logger.info("=" * 60)
                self.logger.info("Ctrl+C 감지 - 프로그램을 종료합니다...")
                self.logger.info("=" * 60)

        except KeyboardInterrupt:
            self.logger.info("\n")
            self.logger.info("=" * 60)
            self.logger.info("사용자 중단 (Ctrl+C)")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"애플리케이션 오류: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self):
        """
        애플리케이션 종료 (Graceful Shutdown)

        정리 순서:
        1. 카메라 루프 중지
        2. 카메라 루프 스레드 종료 대기
        3. ZED 카메라 종료
        4. 측정 기록 CSV 자동 저장
        5. 요약 통계 출력
        """
        # ========== 중복 호출 방지 ==========
        if self.shutdown_called:
            return
        self.shutdown_called = True

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("애플리케이션 종료 중...")
        self.logger.info("=" * 60)

        # ========== 카메라 루프 중지 ==========
        self.running.clear()  # 실행 플래그 OFF → 카메라 루프 종료

        # ========== 카메라 루프 스레드 종료 대기 ==========
        if self.camera_thread and self.camera_thread.is_alive():
            self.logger.info("카메라 루프 스레드 종료 대기 중...")
            self.camera_thread.join(timeout=2.0)  # 최대 2초 대기
            if self.camera_thread.is_alive():
                self.logger.warning("카메라 루프 스레드가 제시간에 종료되지 않았습니다")
            else:
                self.logger.info("카메라 루프 종료")

        self.logger.info("✓ 카메라 루프 중지")

        # ========== 카메라 종료 ==========
        if self.camera:
            self.camera.close()
            self.logger.info("✓ ZED 카메라 종료")

        # ========== CSV 자동 저장 ==========
        # 세션 디렉토리에 CSV 저장
        if len(self.analyzer.records) > 0:
            try:
                csv_filename = f"depth_val_{self.session_timestamp}.csv"
                csv_path = self.analyzer.save_to_csv(self.session_dir / csv_filename)
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info("📊 측정 기록 자동 저장 완료!")
                self.logger.info("=" * 60)
                self.logger.info(f"📁 CSV 파일: {csv_path}")

                # 요약 통계 출력
                summary = self.analyzer.get_summary()
                self.logger.info("")
                self.logger.info("📈 측정 요약:")
                self.logger.info(f"  • 총 측정 횟수: {summary['total_measurements']}")
                self.logger.info(f"  • 평균 오차율: {summary['mean_error_pct']:.2f}%")
                self.logger.info(f"  • 오차율 표준편차: {summary['std_error_pct']:.2f}%")
                self.logger.info(f"  • 최소 오차율: {summary['min_error_pct']:.2f}%")
                self.logger.info(f"  • 최대 오차율: {summary['max_error_pct']:.2f}%")
                self.logger.info("=" * 60)

            except Exception as e:
                self.logger.error(f"❌ CSV 저장 실패: {e}")
        else:
            self.logger.info("ℹ️  저장된 측정 기록이 없습니다")

        self.logger.info("")
        self.logger.info("✅ 종료 완료")
        self.logger.info("=" * 60)

        # ========== Daemon 스레드 정리 대기 ==========
        # Flask 및 기타 daemon 스레드가 ZED SDK 리소스 접근을 완료하도록 짧은 지연
        import time
        time.sleep(0.5)


def main():
    """
    메인 함수 (프로그램 진입점)

    명령행 인자를 파싱하고 애플리케이션 실행
    """
    # ========== 명령행 인자 파싱 ==========
    parser = argparse.ArgumentParser(
        description='ZED ArUco Depth Estimation - 웹 스트리밍 버전'
    )
    parser.add_argument(
        '-c', '--config',
        default='config/config.yaml',
        help='설정 파일 경로 (기본값: config/config.yaml)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        help='웹 서버 포트 (설정 파일 값 오버라이드)'
    )
    parser.add_argument(
        '-H', '--host',
        help='웹 서버 호스트 (설정 파일 값 오버라이드)'
    )

    args = parser.parse_args()

    # ========== 애플리케이션 생성 및 실행 ==========
    app = DepthEstimationApp(config_path=args.config)

    # 명령행 인자로 설정 오버라이드
    if args.port:
        app.config._config.setdefault('web_server', {})['port'] = args.port
    if args.host:
        app.config._config.setdefault('web_server', {})['host'] = args.host

    app.run()


if __name__ == '__main__':
    main()
