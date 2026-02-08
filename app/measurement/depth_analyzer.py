"""
Depth Analysis Module
=====================
ZED 카메라의 Depth 맵을 분석하고 통계를 계산하는 모듈

주요 기능:
1. 마커 영역 Depth 통계 계산 (중앙값, 평균, 표준편차 등)
2. IQR 기반 아웃라이어 제거로 노이즈에 강건한 측정
3. 측정 기록 관리 및 CSV 저장
4. 오차율 계산 (실제 거리 vs ZED 측정값)

Depth 측정 방식:
- 마커 영역 모드: 마커 내부 전체 픽셀의 Depth 통계 (권장)
- 윈도우 모드: 마커 중심 주변 작은 윈도우만 사용 (빠름)
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
    """
    Depth 통계 데이터 클래스

    마커 영역의 Depth 값들을 분석한 결과를 담음.

    Attributes:
        median_mm: 중앙값 (mm) - 가장 대표적인 값, 아웃라이어에 강건
        mean_mm: 평균값 (mm)
        std_mm: 표준편차 (mm) - 값이 작을수록 안정적
        min_mm: 최소값 (mm)
        max_mm: 최대값 (mm)
        raw_valid_ratio: 원본 유효 픽셀 비율 (0~1) - IQR 필터링 전
        filtered_ratio: 필터링 후 유효 픽셀 비율 (0~1) - IQR 필터링 후
        filtered_count: 필터링 후 남은 픽셀 수
        outliers_removed: 제거된 아웃라이어 수
        confidence_mean: 평균 신뢰도 (0~100) - ZED가 depth 값을 얼마나 신뢰하는지
    """
    median_mm: float
    mean_mm: float
    std_mm: float
    min_mm: float
    max_mm: float
    raw_valid_ratio: float
    filtered_ratio: float
    filtered_count: int
    outliers_removed: int
    confidence_mean: float


@dataclass
class MeasurementRecord:
    """
    측정 기록 데이터 클래스

    한 번의 프레임 저장 시 기록되는 모든 정보.
    CSV 파일의 한 행에 해당.

    Attributes:
        image_filename: 저장된 이미지 파일명
        timestamp: 측정 시각 (YYYY-MM-DD HH:MM:SS.ffffff)
        actual_m: 실제 거리 (m) - 레이저 측정기로 측정한 값
        marker_size_mm: 사용된 마커 크기 (mm)
        marker_center_x: 마커 중심 X 좌표 (픽셀)
        marker_center_y: 마커 중심 Y 좌표 (픽셀)
        marker_angle: 마커 회전 각도 (도, -180~180)
        zed_median_mm: ZED Depth 중앙값 (mm)
        zed_mean_mm: ZED Depth 평균값 (mm)
        zed_std_mm: ZED Depth 표준편차 (mm)
        error_mm: 오차 (mm) = |ZED - 실제|
        error_pct: 오차율 (%) = 오차 / 실제 × 100
        raw_valid_ratio: 원본 유효 픽셀 비율 (IQR 필터링 전)
        filtered_ratio: 필터링 후 유효 픽셀 비율 (IQR 필터링 후)
        confidence_mean: 평균 신뢰도 (0~100) - ZED가 depth 값을 얼마나 신뢰하는지
    """
    image_filename: str
    timestamp: str
    actual_m: float
    marker_size_mm: int
    marker_center_x: int
    marker_center_y: int
    marker_angle: float
    zed_median_mm: float
    zed_mean_mm: float
    zed_std_mm: float
    error_mm: float
    error_pct: float
    raw_valid_ratio: float
    filtered_ratio: float
    confidence_mean: float


class DepthAnalyzer:
    """
    Depth 분석 클래스

    마커 영역의 Depth 맵을 분석하여 통계를 계산하고,
    측정 기록을 관리하며 CSV로 저장.

    사용 예시:
        analyzer = DepthAnalyzer(config)
        analyzer.set_distance(1.5)  # 실제 거리 설정 (1.5m)
        stats = analyzer.analyze(depth_map, detection)
        analyzer.add_measurement(stats, detection)
        analyzer.save_to_csv()

    Attributes:
        records: 측정 기록 리스트
        current_distance_m: 현재 설정된 실제 거리 (m)
    """

    def __init__(self, config: Config):
        """
        Depth 분석기 초기화

        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.measurement')

        # 측정 기록 저장 리스트
        self.records: List[MeasurementRecord] = []

        # 현재 측정 거리 (레이저 측정기로 측정한 실제 거리)
        self.current_distance_m: Optional[float] = None

    def analyze_marker_region(
        self,
        depth_map: np.ndarray,
        detection: MarkerDetection,
        confidence_map: Optional[np.ndarray] = None
    ) -> Optional[DepthStats]:
        """
        마커 영역 전체의 Depth 통계 계산 (권장 방식)

        마커 내부 모든 픽셀의 Depth 값을 수집하고,
        IQR 기반 아웃라이어 제거 후 통계 계산.

        처리 흐름:
        1. 마커 영역 마스크 생성 (다각형 채우기)
        2. 유효한 Depth 값 추출 (inf, nan, 0 제외)
        3. IQR 기반 아웃라이어 제거
        4. 통계 계산 (중앙값, 평균, 표준편차 등)

        Args:
            depth_map: Depth 맵 (shape: H x W, 단위: mm)
            detection: 마커 감지 결과 (corners 사용)

        Returns:
            DepthStats: Depth 통계 (유효한 값이 없으면 None)
        """
        h, w = depth_map.shape

        # ========== 1단계: 마커 영역 마스크 생성 ==========
        # 마커 4개 코너로 다각형 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # ========== 2단계: 유효한 Depth 값 추출 ==========
        # 조건: 마스크 내부 AND 유한한 값 AND 양수
        valid_mask = (mask > 0) & np.isfinite(depth_map) & (depth_map > 0)
        valid_values = depth_map[valid_mask]

        if len(valid_values) == 0:
            self.logger.warning("유효한 depth 값이 없습니다")
            return None

        # ========== 3단계: IQR 기반 아웃라이어 제거 ==========
        # IQR (Interquartile Range) = Q3 - Q1
        # 아웃라이어: Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_values = valid_values[
            (valid_values >= lower_bound) & (valid_values <= upper_bound)
        ]

        # 필터링 결과가 비어있으면 원본 사용 (fallback)
        if len(filtered_values) > 0:
            use_values = filtered_values
        else:
            # IQR 필터링으로 모든 값이 제거됨 - 원본으로 폴백
            self.logger.warning(
                f"IQR 필터링 결과가 비어있어 원본 값 사용 (원본 개수: {len(valid_values)})"
            )
            use_values = valid_values

        # ========== 4단계: 통계 계산 ==========
        # 유효 비율 계산
        total_mask_pixels = np.sum(mask > 0)
        raw_valid_ratio = float(len(valid_values) / total_mask_pixels)
        filtered_ratio = float(len(use_values) / total_mask_pixels)

        median_raw = float(np.median(use_values))

        # 디버깅용 로그
        self.logger.debug(
            f"Raw depth 값 (mm): median={median_raw:.2f}, "
            f"min={np.min(use_values):.2f}, max={np.max(use_values):.2f}"
        )

        # ========== Confidence 계산 ==========
        # confidence_map이 제공된 경우 동일한 마스크 영역에서 평균 신뢰도 계산
        if confidence_map is not None:
            # 마스크 내부의 유효한 confidence 값 추출
            confidence_valid_mask = (mask > 0) & np.isfinite(confidence_map) & (confidence_map >= 0)
            confidence_values = confidence_map[confidence_valid_mask]

            if len(confidence_values) > 0:
                confidence_mean = float(np.mean(confidence_values))
            else:
                confidence_mean = 0.0
        else:
            confidence_mean = 0.0

        stats = DepthStats(
            median_mm=median_raw,                              # 중앙값 (가장 중요)
            mean_mm=float(np.mean(use_values)),               # 평균
            std_mm=float(np.std(use_values)),                 # 표준편차
            min_mm=float(np.min(use_values)),                 # 최소
            max_mm=float(np.max(use_values)),                 # 최대
            raw_valid_ratio=raw_valid_ratio,                  # IQR 필터링 전 유효 비율
            filtered_ratio=filtered_ratio,                     # IQR 필터링 후 유효 비율
            filtered_count=len(use_values),                    # 필터 후 개수
            outliers_removed=len(valid_values) - len(use_values),  # 제거된 아웃라이어 수
            confidence_mean=confidence_mean                    # 평균 신뢰도 (0~100)
        )

        return stats

    def analyze_window(
        self,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        window_size: Optional[int] = None,
        confidence_map: Optional[np.ndarray] = None
    ) -> Optional[DepthStats]:
        """
        중심점 주변 윈도우의 Depth 통계 계산 (빠른 방식)

        마커 전체가 아닌 중심점 주변 작은 영역만 분석.
        빠르지만 마커 영역 방식보다 덜 정확함.

        Note:
            윈도우 모드는 속도를 위해 IQR 기반 아웃라이어 제거를 적용하지 않습니다.
            정확도가 중요한 실험에서는 마커 영역 모드(analyze_marker_region)를 사용하세요.

        Args:
            depth_map: Depth 맵 (shape: H x W, 단위: mm)
            center_x: 중심점 X 좌표
            center_y: 중심점 Y 좌표
            window_size: 윈도우 크기 (None이면 설정값 사용)

        Returns:
            DepthStats: Depth 통계 (유효한 값이 없으면 None)
        """
        if window_size is None:
            window_size = self.config.window_size

        half = window_size // 2
        h, w = depth_map.shape

        # 윈도우 영역 추출 (경계 처리 포함)
        y1, y2 = max(0, center_y - half), min(h, center_y + half + 1)
        x1, x2 = max(0, center_x - half), min(w, center_x + half + 1)
        window = depth_map[y1:y2, x1:x2]

        # 유효한 값 추출
        valid_mask = np.isfinite(window) & (window > 0)
        valid_values = window[valid_mask]

        if len(valid_values) == 0:
            self.logger.warning("유효한 depth 값이 없습니다")
            return None

        # 통계 계산 (윈도우 방식은 아웃라이어 제거 안함 - 속도 우선)
        valid_ratio = float(len(valid_values) / window.size)

        # Confidence 계산 (윈도우 영역)
        if confidence_map is not None:
            confidence_window = confidence_map[y1:y2, x1:x2]
            confidence_valid_values = confidence_window[np.isfinite(confidence_window) & (confidence_window >= 0)]
            confidence_mean = float(np.mean(confidence_valid_values)) if len(confidence_valid_values) > 0 else 0.0
        else:
            confidence_mean = 0.0

        stats = DepthStats(
            median_mm=float(np.median(valid_values)),
            mean_mm=float(np.mean(valid_values)),
            std_mm=float(np.std(valid_values)),
            min_mm=float(np.min(valid_values)),
            max_mm=float(np.max(valid_values)),
            raw_valid_ratio=valid_ratio,      # 윈도우 모드는 필터링 없음
            filtered_ratio=valid_ratio,        # raw와 동일 (아웃라이어 제거 없음)
            filtered_count=len(valid_values),
            outliers_removed=0,  # 윈도우 방식은 아웃라이어 제거 안함
            confidence_mean=confidence_mean  # 평균 신뢰도
        )

        return stats

    def analyze(
        self,
        depth_map: np.ndarray,
        detection: MarkerDetection,
        confidence_map: Optional[np.ndarray] = None
    ) -> Optional[DepthStats]:
        """
        설정에 따라 적절한 방식으로 Depth 분석

        config.use_marker_region에 따라:
        - True: analyze_marker_region() 호출
        - False: analyze_window() 호출

        Args:
            depth_map: Depth 맵
            detection: 마커 감지 결과
            confidence_map: Confidence 맵 (선택사항)

        Returns:
            DepthStats: Depth 통계
        """
        if self.config.use_marker_region:
            return self.analyze_marker_region(depth_map, detection, confidence_map)
        else:
            return self.analyze_window(
                depth_map,
                detection.center_x,
                detection.center_y,
                confidence_map=confidence_map
            )

    def add_measurement(
        self,
        stats: DepthStats,
        detection: MarkerDetection,
        actual_distance_m: Optional[float] = None,
        marker_size_mm: Optional[int] = None,
        image_filename: Optional[str] = None
    ) -> bool:
        """
        측정 기록 추가

        Depth 통계와 마커 정보를 결합하여 측정 레코드 생성.
        오차 계산: |ZED측정값 - 실제값| / 실제값 × 100

        Args:
            stats: Depth 통계
            detection: 마커 감지 결과
            actual_distance_m: 실제 거리 (m), None이면 현재 설정값 사용
            marker_size_mm: 사용된 마커 크기 (mm)
            image_filename: 저장된 이미지 파일명

        Returns:
            bool: 성공 여부 (실제 거리 미설정 시 False)
        """
        # 실제 거리가 지정되지 않으면 현재 설정값 사용
        if actual_distance_m is None:
            actual_distance_m = self.current_distance_m

        if actual_distance_m is None:
            self.logger.warning("실제 거리가 설정되지 않았습니다")
            return False

        # 마커 크기가 지정되지 않으면 0으로 기록 (경고)
        if marker_size_mm is None:
            self.logger.warning("마커 크기가 지정되지 않았습니다")
            marker_size_mm = 0

        # ===== 오차 계산 =====
        # NOTE: 오차 계산에 median을 사용하는 이유:
        # - median은 아웃라이어에 강건(robust)하여 노이즈가 많은 depth 데이터에 적합
        # - mean은 극단값에 영향을 받아 실제 대표값을 왜곡할 수 있음
        # - 연구 논문에서도 depth 정확도 평가 시 median 사용이 일반적
        actual_mm = actual_distance_m * 1000  # m → mm
        error_mm = abs(stats.median_mm - actual_mm)
        error_pct = (error_mm / actual_mm) * 100

        # ===== 기록 생성 =====
        record = MeasurementRecord(
            image_filename=image_filename or "N/A",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            actual_m=actual_distance_m,
            marker_size_mm=marker_size_mm,
            marker_center_x=detection.center_x,
            marker_center_y=detection.center_y,
            marker_angle=detection.angle,
            zed_median_mm=stats.median_mm,
            zed_mean_mm=stats.mean_mm,
            zed_std_mm=stats.std_mm,
            error_mm=error_mm,
            error_pct=error_pct,
            raw_valid_ratio=stats.raw_valid_ratio,
            filtered_ratio=stats.filtered_ratio,
            confidence_mean=stats.confidence_mean
        )

        self.records.append(record)

        self.logger.info(
            f"측정 추가: 실제={actual_distance_m:.3f}m, "
            f"ZED={stats.median_mm:.1f}mm, 오차={error_pct:.2f}%"
        )

        return True

    def set_distance(self, distance_m: float) -> None:
        """
        현재 측정 거리 설정

        레이저 측정기로 측정한 실제 거리를 입력.
        이후 add_measurement() 호출 시 이 값이 기준이 됨.

        Args:
            distance_m: 실제 거리 (미터 단위)
        """
        self.current_distance_m = distance_m
        self.logger.info(f"측정 거리 설정: {distance_m:.3f}m")

    def save_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        측정 기록을 CSV 파일로 저장

        파일명 형식: depth_val_YYYYMMDD_HHMMSS.csv

        Args:
            output_path: 출력 파일 경로 (None이면 자동 생성)

        Returns:
            Path: 저장된 파일 경로

        Raises:
            ValueError: 저장할 기록이 없는 경우
        """
        if not self.records:
            raise ValueError("저장할 측정 기록이 없습니다")

        # 출력 경로 자동 생성
        if output_path is None:
            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"depth_val_{timestamp}.csv"

        # CSV 저장 (float 값은 소수점 4자리로 포맷팅 - 연구 논문용 가독성)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # 헤더는 dataclass 필드명 사용
            fieldnames = list(asdict(self.records[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # float 값 포맷팅 (소수점 4자리)
            for record in self.records:
                row = asdict(record)
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted_row[key] = f"{value:.4f}"
                    else:
                        formatted_row[key] = value
                writer.writerow(formatted_row)

        self.logger.info(f"CSV 저장 완료: {output_path} ({len(self.records)} 기록)")
        return output_path

    def get_error_color(self, error_pct: float) -> tuple:
        """
        오차율에 따른 표시 색상 반환

        색상 기준:
        - < 2%: 녹색 (매우 정확)
        - 2~5%: 노란색 (양호)
        - 5~10%: 주황색 (보통)
        - > 10%: 빨간색 (부정확)

        Args:
            error_pct: 오차율 (%)

        Returns:
            tuple: BGR 색상 (OpenCV 형식)
        """
        if np.isnan(error_pct):
            return (128, 128, 128)  # 회색 (계산 불가)
        if error_pct < 2.0:
            return (0, 255, 0)      # 녹색
        if error_pct < 5.0:
            return (0, 255, 255)    # 노란색 (BGR)
        if error_pct < 10.0:
            return (0, 165, 255)    # 주황색 (BGR)
        return (0, 0, 255)          # 빨간색

    def get_summary(self) -> dict:
        """
        측정 기록 요약 통계 반환

        Returns:
            dict: 요약 통계
                - total_measurements: 총 측정 횟수
                - mean_error_pct: 평균 오차율
                - std_error_pct: 오차율 표준편차
                - min_error_pct: 최소 오차율
                - max_error_pct: 최대 오차율
        """
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
