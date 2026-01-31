"""
ArUco marker detection module
ArUco 마커를 감지하고 검증합니다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import Config


class MarkerValidationError(Exception):
    """마커 검증 에러"""
    pass


@dataclass
class MarkerDetection:
    """마커 감지 결과"""
    marker_id: int
    center_x: int
    center_y: int
    corners: np.ndarray
    angle: float
    area: float
    is_valid: bool
    validation_message: str


class ArucoDetector:
    """ArUco 마커 감지 클래스"""

    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.marker')
        self.target_marker_id = config.marker_id

        # ArUco dictionary 초기화
        try:
            aruco_dict_type = getattr(cv2.aruco, config.aruco_dict)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        except AttributeError:
            raise ValueError(f"잘못된 ArUco dictionary: {config.aruco_dict}")

        # Detector 초기화 (OpenCV 버전에 따라 다름)
        try:
            # OpenCV 4.7.0+
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.use_new_api = True
            self.logger.info("ArUco Detector 초기화 (신규 API)")
        except AttributeError:
            # OpenCV 4.5.x 이하
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
            self.logger.info("ArUco Detector 초기화 (레거시 API)")

        self.logger.info(f"타겟 마커 ID: {self.target_marker_id}")
        self.logger.info(f"ArUco Dictionary: {config.aruco_dict}")

    def detect(self, image: np.ndarray) -> Optional[MarkerDetection]:
        """
        이미지에서 타겟 마커를 감지합니다.

        Args:
            image: 입력 이미지 (BGR 또는 RGB)

        Returns:
            마커 감지 결과 또는 None (마커가 없는 경우)
        """
        # 마커 감지
        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(image)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, self.aruco_dict, parameters=self.parameters
            )

        # 타겟 마커 찾기
        if ids is not None:
            for i, detected_id in enumerate(ids.flatten()):
                if detected_id == self.target_marker_id:
                    marker_corners = corners[i][0]

                    # 마커 검증
                    is_valid, msg = self._validate_marker(marker_corners, image.shape)

                    # 중심점 계산
                    center = marker_corners.mean(axis=0)
                    cx, cy = int(center[0]), int(center[1])

                    # 각도 및 면적 계산
                    angle = self._calculate_angle(marker_corners)
                    area = cv2.contourArea(marker_corners.astype(np.float32))

                    detection = MarkerDetection(
                        marker_id=detected_id,
                        center_x=cx,
                        center_y=cy,
                        corners=marker_corners,
                        angle=angle,
                        area=area,
                        is_valid=is_valid,
                        validation_message=msg
                    )

                    if not is_valid:
                        self.logger.warning(f"마커 검증 실패: {msg}")

                    return detection

        return None

    def _validate_marker(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, ...]
    ) -> Tuple[bool, str]:
        """
        마커가 유효한지 검증합니다.

        Args:
            corners: 마커 코너 좌표
            image_shape: 이미지 shape

        Returns:
            (검증 성공 여부, 메시지)
        """
        h, w = image_shape[:2]
        margin = 10

        # 경계 체크
        if (np.any(corners < margin) or
            np.any(corners[:, 0] > w - margin) or
            np.any(corners[:, 1] > h - margin)):
            return False, "마커가 이미지 경계에 너무 가까움"

        # 면적 체크
        area = cv2.contourArea(corners.astype(np.float32))
        if area < 400:
            return False, f"마커가 너무 작음 (면적: {area:.0f}px)"

        # 마커 크기 체크 (너무 크면 너무 가까움)
        if area > (w * h * 0.3):
            return False, f"마커가 너무 큼 (면적: {area:.0f}px)"

        return True, "정상"

    def _calculate_angle(self, corners: np.ndarray) -> float:
        """
        마커의 기울기 각도를 계산합니다.

        Args:
            corners: 마커 코너 좌표

        Returns:
            각도 (degrees)
        """
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]

        width = np.linalg.norm(v1)
        height = np.linalg.norm(v2)

        if max(width, height) == 0:
            return 0.0

        # 가로세로 비율로 각도 추정
        aspect_ratio = min(width, height) / max(width, height)
        angle_estimate = np.arccos(np.clip(aspect_ratio, 0, 1)) * 180 / np.pi

        return angle_estimate

    def draw_marker(
        self,
        image: np.ndarray,
        detection: MarkerDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> None:
        """
        이미지에 마커를 그립니다.

        Args:
            image: 입력 이미지 (수정됨)
            detection: 마커 감지 결과
            color: 그리기 색상 (BGR)
            thickness: 선 두께
        """
        # 마커 영역 채우기 (반투명)
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # 마커 외곽선
        cv2.polylines(image, [pts], True, color, thickness)

        # 중심점
        cv2.circle(
            image,
            (detection.center_x, detection.center_y),
            8,
            color,
            -1
        )

        # 마커 ID 표시
        cv2.putText(
            image,
            f"ID: {detection.marker_id}",
            (detection.center_x + 20, detection.center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4
        )
        cv2.putText(
            image,
            f"ID: {detection.marker_id}",
            (detection.center_x + 20, detection.center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
