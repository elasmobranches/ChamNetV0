"""
Depth analysis module
Depth 맵을 분석하고 통계를 계산합니다.
"""

import cv2
import numpy as np
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from ..utils.logger import get_logger
from ..utils.config import Config
from ..marker.aruco_detector import MarkerDetection


@dataclass
class DepthStats:
    """Depth 통계"""
    median_mm: float
    mean_mm: float
    std_mm: float
    min_mm: float
    max_mm: float
    valid_ratio: float
    filtered_count: int
    outliers_removed: int


@dataclass
class MeasurementRecord:
    """측정 기록"""
    image_filename: str
    timestamp: str
    actual_m: float
    zed_median_mm: float
    zed_mean_mm: float
    zed_std_mm: float
    error_mm: float
    error_pct: float
    marker_angle: float
    valid_ratio: float


class DepthAnalyzer:
    """Depth 분석 클래스"""

    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.measurement')

        # 측정 기록
        self.records: List[MeasurementRecord] = []

        # 현재 측정 거리
        self.current_distance_m: Optional[float] = None

    def analyze_marker_region(
        self,
        depth_map: np.ndarray,
        detection: MarkerDetection
    ) -> Optional[DepthStats]:
        """
        마커 영역 전체의 depth 통계를 계산합니다.

        Args:
            depth_map: Depth 맵 (mm 단위)
            detection: 마커 감지 결과

        Returns:
            Depth 통계 또는 None (유효한 값이 없는 경우)
        """
        h, w = depth_map.shape

        # 마커 영역 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # 유효한 depth 값 추출
        valid_mask = (mask > 0) & np.isfinite(depth_map) & (depth_map > 0)
        valid_values = depth_map[valid_mask]

        if len(valid_values) == 0:
            self.logger.warning("유효한 depth 값이 없습니다")
            return None

        # IQR 기반 아웃라이어 제거
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_values = valid_values[
            (valid_values >= lower_bound) & (valid_values <= upper_bound)
        ]

        # 필터링된 값이 없으면 원본 사용
        use_values = filtered_values if len(filtered_values) > 0 else valid_values

        # 통계 계산
        median_raw = float(np.median(use_values))

        # 디버깅: Raw depth 값 로깅 (mm 단위)
        self.logger.debug(f"Raw depth 값 (mm): median={median_raw:.2f}, min={np.min(use_values):.2f}, max={np.max(use_values):.2f}")

        stats = DepthStats(
            median_mm=median_raw,  # ZED SDK가 MILLIMETER로 반환
            mean_mm=float(np.mean(use_values)),
            std_mm=float(np.std(use_values)),
            min_mm=float(np.min(use_values)),
            max_mm=float(np.max(use_values)),
            valid_ratio=float(len(valid_values) / np.sum(mask > 0)),
            filtered_count=len(use_values),
            outliers_removed=len(valid_values) - len(use_values)
        )

        return stats

    def analyze_window(
        self,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        window_size: Optional[int] = None
    ) -> Optional[DepthStats]:
        """
        중심점 주변 윈도우의 depth 통계를 계산합니다.

        Args:
            depth_map: Depth 맵 (mm 단위)
            center_x: 중심점 X 좌표
            center_y: 중심점 Y 좌표
            window_size: 윈도우 크기 (None이면 설정값 사용)

        Returns:
            Depth 통계 또는 None (유효한 값이 없는 경우)
        """
        if window_size is None:
            window_size = self.config.window_size

        half = window_size // 2
        h, w = depth_map.shape

        # 윈도우 영역 추출
        y1, y2 = max(0, center_y - half), min(h, center_y + half + 1)
        x1, x2 = max(0, center_x - half), min(w, center_x + half + 1)
        window = depth_map[y1:y2, x1:x2]

        # 유효한 값 추출
        valid_values = window[np.isfinite(window) & (window > 0)]

        if len(valid_values) == 0:
            self.logger.warning("유효한 depth 값이 없습니다")
            return None

        # 통계 계산 (윈도우 방식은 아웃라이어 제거 안함)
        stats = DepthStats(
            median_mm=float(np.median(valid_values)),
            mean_mm=float(np.mean(valid_values)),
            std_mm=float(np.std(valid_values)),
            min_mm=float(np.min(valid_values)),
            max_mm=float(np.max(valid_values)),
            valid_ratio=float(len(valid_values) / window.size),
            filtered_count=len(valid_values),
            outliers_removed=0
        )

        return stats

    def analyze(
        self,
        depth_map: np.ndarray,
        detection: MarkerDetection
    ) -> Optional[DepthStats]:
        """
        설정에 따라 적절한 방식으로 depth를 분석합니다.

        Args:
            depth_map: Depth 맵
            detection: 마커 감지 결과

        Returns:
            Depth 통계
        """
        if self.config.use_marker_region:
            return self.analyze_marker_region(depth_map, detection)
        else:
            return self.analyze_window(
                depth_map,
                detection.center_x,
                detection.center_y
            )

    def add_measurement(
        self,
        stats: DepthStats,
        detection: MarkerDetection,
        actual_distance_m: Optional[float] = None,
        image_filename: Optional[str] = None
    ) -> bool:
        """
        측정 기록을 추가합니다.

        Args:
            stats: Depth 통계
            detection: 마커 감지 결과
            actual_distance_m: 실제 거리 (m)
            image_filename: 저장된 이미지 파일 이름

        Returns:
            성공 여부
        """
        if actual_distance_m is None:
            actual_distance_m = self.current_distance_m

        if actual_distance_m is None:
            self.logger.warning("실제 거리가 설정되지 않았습니다")
            return False

        # 오차 계산
        actual_mm = actual_distance_m * 1000
        error_mm = abs(stats.median_mm - actual_mm)
        error_pct = (error_mm / actual_mm) * 100

        # 기록 생성
        record = MeasurementRecord(
            image_filename=image_filename or "N/A",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            actual_m=actual_distance_m,
            zed_median_mm=stats.median_mm,
            zed_mean_mm=stats.mean_mm,
            zed_std_mm=stats.std_mm,
            error_mm=error_mm,
            error_pct=error_pct,
            marker_angle=detection.angle,
            valid_ratio=stats.valid_ratio
        )

        self.records.append(record)
        self.logger.info(
            f"측정 추가: 실제={actual_distance_m:.3f}m, "
            f"ZED={stats.median_mm:.1f}mm, 오차={error_pct:.2f}%"
        )

        return True

    def set_distance(self, distance_m: float) -> None:
        """현재 측정 거리를 설정합니다."""
        self.current_distance_m = distance_m
        self.logger.info(f"측정 거리 설정: {distance_m:.3f}m")

    def save_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        측정 기록을 CSV 파일로 저장합니다.

        Args:
            output_path: 출력 파일 경로 (None이면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if not self.records:
            raise ValueError("저장할 측정 기록이 없습니다")

        if output_path is None:
            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"depth_val_{timestamp}.csv"

        # CSV 저장
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(asdict(self.records[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(r) for r in self.records])

        self.logger.info(f"CSV 저장 완료: {output_path} ({len(self.records)} 기록)")
        return output_path

    def get_error_color(self, error_pct: float) -> tuple:
        """
        오차율에 따른 색상을 반환합니다.

        Args:
            error_pct: 오차율 (%)

        Returns:
            BGR 색상 튜플
        """
        if np.isnan(error_pct):
            return (128, 128, 128)  # Gray
        if error_pct < 2.0:
            return (0, 255, 0)      # Green
        if error_pct < 5.0:
            return (0, 255, 255)    # Yellow
        if error_pct < 10.0:
            return (0, 165, 255)    # Orange
        return (0, 0, 255)          # Red

    def get_summary(self) -> dict:
        """측정 기록 요약을 반환합니다."""
        if not self.records:
            return {
                'total_measurements': 0,
                'mean_error_pct': 0,
                'std_error_pct': 0,
                'min_error_pct': 0,
                'max_error_pct': 0
            }

        errors = [r.error_pct for r in self.records]

        return {
            'total_measurements': len(self.records),
            'mean_error_pct': np.mean(errors),
            'std_error_pct': np.std(errors),
            'min_error_pct': np.min(errors),
            'max_error_pct': np.max(errors)
        }
