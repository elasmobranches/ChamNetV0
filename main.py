#!/usr/bin/env python3
"""
ZED ArUco Depth Estimation - Main Application
ZED 2i 카메라와 ArUco 마커를 이용한 거리 측정 시스템 (Flask 웹 스트리밍 버전)
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
    """메인 애플리케이션 클래스"""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        try:
            self.config = Config(config_path)
        except ConfigError as e:
            print(f"❌ 설정 로드 실패: {e}")
            sys.exit(1)

        # 로거 초기화
        self.logger = setup_logger(
            name='depth_estimation',
            log_dir=self.config.log_dir,
            level='INFO'
        )

        self.logger.info("=" * 60)
        self.logger.info("ZED ArUco Depth Estimation v2.0 시작")
        self.logger.info("=" * 60)

        # 컴포넌트 초기화
        self.camera = ZEDCamera(self.config)
        self.detector = ArucoDetector(self.config)
        self.analyzer = DepthAnalyzer(self.config)
        self.server = StreamServer(self.config)

        # 웹 서버 콜백 등록
        self.server.on_distance_input = self._on_distance_input
        self.server.on_save_frame = self._on_save_frame
        self.server.on_toggle_mode = self._on_toggle_mode

        # 상태 관리
        self.running = Event()
        self.state_lock = Lock()  # 공유 상태 보호용
        self.current_frame_display = None
        self.current_detection = None
        self.current_depth_stats = None

        # 종료 시그널 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info("종료 시그널 수신")
        self.shutdown()

    def _on_distance_input(self, distance: float):
        """거리 입력 콜백"""
        self.analyzer.set_distance(distance)

    def _on_save_frame(self):
        """프레임 저장 콜백"""
        # Lock으로 보호된 상태 읽기
        with self.state_lock:
            # 조건 체크
            if not self.current_detection:
                raise ValueError("마커가 감지되지 않았습니다. 마커를 카메라에 보이도록 배치하세요.")

            if not self.current_depth_stats:
                raise ValueError("Depth 데이터가 없습니다.")

            if self.analyzer.current_distance_m is None:
                raise ValueError("거리가 설정되지 않았습니다. 먼저 거리를 입력하세요 (D 키).")

            if self.current_frame_display is None:
                raise ValueError("프레임 데이터가 없습니다.")

            # 복사본 생성 (lock 밖에서 사용하기 위해)
            frame_display = self.current_frame_display.copy()
            depth_stats = self.current_depth_stats
            detection = self.current_detection

        # 이미지 파일 이름 생성
        distance = self.analyzer.current_distance_m
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{distance:.3f}m_{timestamp}.jpg"

        # 이미지 저장
        output_dir = self.config.output_dir / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        success = cv2.imwrite(str(filepath), frame_display)
        if not success:
            raise IOError(f"이미지 저장 실패: {filepath}")

        # 측정 기록 추가 (이미지 파일 이름 포함)
        success = self.analyzer.add_measurement(
            depth_stats,
            detection,
            image_filename=filename
        )

        if not success:
            raise RuntimeError("측정 기록 추가에 실패했습니다.")

        self.logger.info(f"✓ 프레임 저장 성공: {filepath}")

    def _on_toggle_mode(self):
        """측정 모드 토글 콜백"""
        current = self.config.use_marker_region
        new_mode = not current
        self.config.use_marker_region = new_mode  # Setter 사용
        mode_text = '마커 영역' if new_mode else '윈도우'
        self.logger.info(f"측정 모드 변경: {mode_text}")

    def _process_frame(self, image: np.ndarray, depth_map: np.ndarray):
        """프레임 처리"""
        display = image.copy()

        # 마커 감지
        detection = self.detector.detect(image)

        marker_detected = False
        depth_stats = None
        error_pct = np.nan

        if detection and detection.is_valid:
            marker_detected = True

            # Depth 분석
            depth_stats = self.analyzer.analyze(depth_map, detection)

            if depth_stats:
                # 오차율 계산
                if self.analyzer.current_distance_m:
                    actual_mm = self.analyzer.current_distance_m * 1000
                    error_mm = abs(depth_stats.median_mm - actual_mm)
                    error_pct = (error_mm / actual_mm) * 100

                # 색상 결정
                color = self.analyzer.get_error_color(error_pct)

                # 마커 그리기
                self._draw_marker_visualization(
                    display, detection, depth_stats, error_pct, color
                )

        # 상태 바 그리기
        self._draw_status_bar(display, marker_detected)

        # Lock으로 보호된 상태 업데이트 (atomic하게 한 번에)
        with self.state_lock:
            self.current_frame_display = display.copy()
            self.current_detection = detection
            self.current_depth_stats = depth_stats

        # 웹 서버에 프레임 전송
        self.server.update_frame(display)

        # 웹 서버 상태 업데이트
        self.server.update_status(
            marker_detected=marker_detected,
            zed_depth=depth_stats.median_mm if depth_stats else None,
            error_pct=error_pct if not np.isnan(error_pct) else None,
            saved_count=len(self.analyzer.records)
        )

    def _draw_marker_visualization(
        self,
        image: np.ndarray,
        detection: MarkerDetection,
        stats: DepthStats,
        error_pct: float,
        color: tuple
    ):
        """마커 시각화"""
        # 마커 영역 그리기
        self.detector.draw_marker(image, detection, color)

        # 정보 텍스트
        cx, cy = detection.center_x, detection.center_y
        zed_m = stats.median_mm / 1000

        text_lines = [
            f"ZED: {zed_m:.3f}m",
            f"Error: {error_pct:.2f}%" if not np.isnan(error_pct) else f"ZED: {zed_m:.3f}m",
            f"Std: {stats.std_mm:.1f}mm | Ang: {detection.angle:.1f}deg"
        ]

        # 텍스트 그리기 (그림자 효과)
        for i, text in enumerate(text_lines):
            y_pos = cy + 60 + i * 30
            # 검은 배경
            cv2.putText(
                image, text, (cx + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5
            )
            # 컬러 텍스트
            cv2.putText(
                image, text, (cx + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    def _draw_status_bar(self, image: np.ndarray, marker_detected: bool):
        """상태 바 그리기"""
        h, w = image.shape[:2]

        # 배경
        cv2.rectangle(image, (0, 0), (w, 90), (40, 40, 40), -1)

        # 거리 정보
        if self.analyzer.current_distance_m:
            dist_text = f"Target: {self.analyzer.current_distance_m:.3f} m"
            color = (0, 255, 0)
        else:
            dist_text = "Target: Not Set"
            color = (0, 255, 255)

        cv2.putText(image, dist_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 통계 정보
        mode_text = "Mode: Marker Region" if self.config.use_marker_region else "Mode: Window"
        saved_text = f"Saved: {len(self.analyzer.records)}"
        marker_text = "Marker: Detected" if marker_detected else "Marker: Not Found"

        cv2.putText(image, mode_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, saved_text, (350, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(
            image, marker_text, (550, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0) if marker_detected else (0, 0, 255), 1
        )

    def _camera_loop(self):
        """카메라 루프 (별도 스레드)"""
        try:
            self.logger.info("카메라 루프 시작")

            while self.running.is_set():
                try:
                    # 프레임 캡처
                    image, depth_map = self.camera.get_frame()

                    # 프레임 처리
                    self._process_frame(image, depth_map)

                    # FPS 제어
                    time.sleep(1.0 / self.config.stream_fps)

                except ZEDCameraError as e:
                    self.logger.error(f"프레임 캡처 실패: {e}")
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"카메라 루프 오류: {e}", exc_info=True)
        finally:
            self.logger.info("카메라 루프 종료")

    def _run_flask_server(self):
        """Flask 서버 실행 (별도 스레드)"""
        try:
            # Flask 로거 레벨 조정
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            # Flask 서버 실행
            self.server.app.run(
                host=self.config.web_host,
                port=self.config.web_port,
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
            self.logger.info("ZED 카메라 초기화 중...")
            self.camera.open()

            # 카메라 루프 시작 (별도 스레드)
            self.running.set()
            camera_thread = Thread(target=self._camera_loop, daemon=True)
            camera_thread.start()

            # Flask 서버 시작 (별도 스레드)
            self.logger.info("Flask 웹 서버 시작...")
            self.logger.info(f"브라우저에서 http://{self.config.web_host}:{self.config.web_port} 접속")
            self.logger.info("종료하려면 Ctrl+C를 누르거나 웹 UI에서 '종료' 버튼을 클릭하세요")

            flask_thread = Thread(target=self._run_flask_server, daemon=True)
            flask_thread.start()

            # 메인 스레드는 종료 신호를 기다림
            try:
                while self.running.is_set() and not self.server.is_shutdown_requested():
                    time.sleep(0.5)
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
        """애플리케이션 종료"""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("애플리케이션 종료 중...")
        self.logger.info("=" * 60)

        # 카메라 루프 중지
        self.running.clear()
        self.logger.info("✓ 카메라 루프 중지")

        # 카메라 종료
        if self.camera:
            self.camera.close()
            self.logger.info("✓ ZED 카메라 종료")

        # CSV 자동 저장
        if len(self.analyzer.records) > 0:
            try:
                csv_path = self.analyzer.save_to_csv()
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info("📊 측정 기록 자동 저장 완료!")
                self.logger.info("=" * 60)
                self.logger.info(f"📁 CSV 파일: {csv_path}")

                # 요약 정보
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


def main():
    """메인 함수"""
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

    # 애플리케이션 실행
    app = DepthEstimationApp(config_path=args.config)

    # 명령행 인자로 설정 오버라이드
    if args.port:
        app.config._config.setdefault('web_server', {})['port'] = args.port
    if args.host:
        app.config._config.setdefault('web_server', {})['host'] = args.host

    app.run()


if __name__ == '__main__':
    main()
