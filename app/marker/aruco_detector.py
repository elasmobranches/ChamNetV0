"""
ArUco Marker Detection Module
=============================
ArUco 마커를 감지하고 위치 정보를 추출하는 모듈

ArUco 마커란?
- 정사각형 이진 패턴으로 구성된 시각적 마커
- 각 마커는 고유한 ID를 가짐
- 컴퓨터 비전에서 위치 추적, 카메라 캘리브레이션 등에 사용

지원하는 ArUco Dictionary:
- DICT_4X4_50: 4x4 비트, 50개 마커 (작은 마커에 적합)
- DICT_5X5_100: 5x5 비트, 100개 마커
- DICT_6X6_250: 6x6 비트, 250개 마커
- DICT_7X7_1000: 7x7 비트, 1000개 마커 (큰 마커에 적합)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import Config


@dataclass
class MarkerDetection:
    """
    마커 감지 결과를 담는 데이터 클래스

    Attributes:
        marker_id: 감지된 마커의 고유 ID
        center_x: 마커 중심점 X 좌표 (픽셀)
        center_y: 마커 중심점 Y 좌표 (픽셀)
        corners: 마커 4개 코너 좌표 (shape: 4x2)
                 순서: 좌상 → 우상 → 우하 → 좌하
        angle: 마커 기울기 각도 (도 단위, 0=정면)
        area: 마커 영역 크기 (픽셀 제곱)
        is_valid: 검증 통과 여부
        validation_message: 검증 결과 메시지
    """
    marker_id: int
    center_x: int
    center_y: int
    corners: np.ndarray
    angle: float
    area: float
    is_valid: bool
    validation_message: str


class ArucoDetector:
    """
    ArUco 마커 감지 클래스

    OpenCV의 ArUco 모듈을 사용하여 이미지에서 마커를 찾고,
    위치, 크기, 각도 등의 정보를 추출.

    사용 예시:
        detector = ArucoDetector(config)ㄹ
        detection = detector.detect(image)
        if detection and detection.is_valid:
            print(f"마커 발견: ID={detection.marker_id}")

    Attributes:
        target_marker_id: 찾을 마커 ID (설정 파일에서 지정)
        aruco_dict: ArUco 딕셔너리
        detector: ArUco 감지기 (OpenCV 버전에 따라 다름)
    """

    def __init__(self, config: Config):
        """
        ArUco 감지기 초기화

        Args:
            config: 설정 객체 (marker_id, aruco_dict 등 포함)

        Raises:
            ValueError: 잘못된 ArUco 딕셔너리 이름
        """
        self.config = config
        self.logger = get_logger('depth_estimation.marker')
        self.target_marker_id = config.marker_id

        # ========== ArUco Dictionary 초기화 ==========
        # config.aruco_dict 예: "DICT_4X4_50"
        try:
            aruco_dict_type = getattr(cv2.aruco, config.aruco_dict)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        except AttributeError:
            raise ValueError(f"잘못된 ArUco dictionary: {config.aruco_dict}")

        # ========== Detector 초기화 ==========
        # OpenCV 버전에 따라 API가 다름
        try:
            # OpenCV 4.7.0 이상: 새로운 API
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.use_new_api = True
            self.logger.info("ArUco Detector 초기화 (신규 API)")
        except AttributeError:
            # OpenCV 4.5.x 이하: 레거시 API
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
            self.logger.info("ArUco Detector 초기화 (레거시 API)")

        # ========== 파라미터 설정 ==========
        self.update_parameters()

        # ========== 시간적 안정화 (Temporal Smoothing) ==========
        # 1-2프레임 순간적으로 놓쳐도 깜빡이지 않도록
        self._last_detection: Optional[MarkerDetection] = None
        self._miss_count: int = 0
        self._max_miss_frames: int = 10  # 3프레임까지는 이전 결과 유지

        self.logger.info(f"타겟 마커 ID: {self.target_marker_id}")
        self.logger.info(f"ArUco Dictionary: {config.aruco_dict}")
        self.logger.info(f"시간적 안정화: {self._max_miss_frames}프레임 버퍼")

    def update_parameters(self) -> None:
        """
        ArUco 파라미터 업데이트
        """
        self.setup_normal_parameters()
        self.logger.debug("일반 모드 파라미터 적용")

        # 신규 API는 detector 재생성 필요
        if self.use_new_api:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def setup_normal_parameters_old(self) -> None:
        """
        일반 환경을 위한 ArUco 파라미터

        철학: 최소한의 설정, 과도한 최적화 금지
        → ArUco는 ROI 트리거 역할, OpenCV 기본값이면 충분
        """
        params = self.parameters

        # 작은 마커 허용 (관대하게)
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0

        # 코너 정제는 기본 수준만 (과도한 정제 금지)
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
    def setup_normal_parameters(self) -> None:
        """
    ArUco 마커 검출 성능 극대화를 위한 파라미터 설정
    환경: 온실 내 조명 변화, 먼 거리의 작은 마커, 나뭇잎에 의한 부분 가림 대응
        """
        params = self.parameters

        # --- 1. 마커 크기 및 거리 (Detection Range) ---
        # 이미지 내에서 마커로 인정할 최소 둘레 비율 (0.01 -> 0.005로 하향)
        # 0.005는 화면 짧은 축의 0.5% 크기만 되어도 인식하겠다는 의미로, 아주 멀리 있는 마커 검출 가능
        params.minMarkerPerimeterRate = 0.005 
    
        # 이미지 내에서 마커로 인정할 최대 둘레 비율
        # 4.0은 마커가 카메라 바로 앞에 위치해 화면을 꽉 채워도 인식 가능함을 의미
        params.maxMarkerPerimeterRate = 4.0

        # --- 3. 코너 정제 (Corner Refinement: 정밀도 향상) ---
        # 검출된 코너를 서브픽셀(소수점 단위) 수준으로 정밀하게 다듬는 방식 설정
        # 위치 추정(Pose Estimation)의 정확도를 높여줌
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


        # --- 4. 에러 복구 및 품질 (Error Correction) ---
        # 마커의 내부 비트가 손상되었을 때 복구할 수 있는 최대 비율
        # 온실에서 잎사귀가 마커를 살짝 가렸을 때 인식률을 높여줌 (너무 높으면 오인식 위험 있음)
        params.errorCorrectionRate = 0.8
    
        # 마커와 이미지 테두리 사이의 최소 거리 (픽셀 단위)
        # 마커가 화면 구석에 걸쳐 있어도 인식되도록 설정
        params.minMarkerDistanceRate = 0.05
    
        # 마커로 최종 판단하기 전, 후보들 간의 최소 거리
        # 높을수록 중복 검출을 방지하고 마커들이 서로 붙어있을 때 구분하게 해줌
        params.minDistanceToBorder = 3

        # --- 5. 마커 비트 추출 (Bit Extraction) ---
        # 마커 내부의 데이터 비트를 읽을 때 사용하는 픽셀 여백
        # 마커가 비스듬하게 보일 때 내부 비트를 더 정확하게 읽도록 도와줌
        params.markerBorderBits = 1

    def detect(self, image: np.ndarray) -> Optional[MarkerDetection]:
        """
        이미지에서 타겟 마커 감지 (시간적 안정화 적용)

        철학: 단순하게 = 안정적으로
        - Normal mode: 원본만 (전처리 없음)
        - Temporal Smoothing: 1-2프레임 순간 미검출은 이전 결과 유지

        Args:
            image: 입력 이미지 (BGR 또는 RGB, shape: H x W x 3)

        Returns:
            MarkerDetection: 마커 감지 결과 (없으면 None)
        """
        # 실제 검출 시도
        result = self._detect_internal(image)

        if result:
            # 검출 성공: 상태 업데이트
            self._last_detection = result
            self._miss_count = 0
            return result
        else:
            # 검출 실패: 시간적 안정화 적용
            self._miss_count += 1

            if self._miss_count <= self._max_miss_frames and self._last_detection:
                # 버퍼 범위 내: 이전 결과 유지 (깜빡임 방지)
                self.logger.debug(
                    f"⏸ 마커 미검출, 이전 결과 유지 ({self._miss_count}/{self._max_miss_frames})"
                )
                return self._last_detection
            else:
                # 버퍼 초과 또는 이전 결과 없음: 실제로 None 반환
                self._last_detection = None
                return None

    def _detect_internal(self, image: np.ndarray) -> Optional[MarkerDetection]:
        """
        내부 검출 로직 (시간적 안정화 없이 순수 검출)

        Args:
            image: 입력 이미지

        Returns:
            MarkerDetection: 마커 감지 결과 (없으면 None)
        """
        # 원본 이미지에서만 감지
        return self._detect_single(image)

    def _detect_single(self, image: np.ndarray) -> Optional[MarkerDetection]:
        """
        단일 이미지에서 마커 감지 시도

        여러 개의 타겟 마커가 감지된 경우, 가장 큰(가까운) 유효한 마커를 선택

        Args:
            image: 입력 이미지

        Returns:
            MarkerDetection: 마커 감지 결과 (없으면 None)
        """
        # ========== [DEBUG] 프레임 밝기 분석 ==========
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness_mean = gray.mean()
        brightness_std = gray.std()
        self.logger.info(f"[밝기 분석] mean={brightness_mean:.1f}, std={brightness_std:.1f}")

        # ========== 마커 감지 ==========
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image, self.aruco_dict, parameters=self.parameters
            )

        # 거부된 후보 로깅 (디버깅용)
        if rejected is not None and len(rejected) > 0:
            self.logger.debug(f"거부된 마커 후보: {len(rejected)}개")

        # ========== 타겟 마커 찾기 (모든 후보 검사) ==========
        if ids is None:
            return None

        self.logger.debug(f"감지된 마커 IDs: {ids.flatten()}, 개수: {len(ids)}")

        # 타겟 ID와 일치하는 모든 후보 수집
        candidates = []

        for i, detected_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            area = cv2.contourArea(marker_corners.astype(np.float32))
            self.logger.debug(f"마커 ID {detected_id}: 면적={area:.0f}px")

            if detected_id == self.target_marker_id:
                # ----- 마커 검증 -----
                is_valid, msg = self._validate_marker(marker_corners, image.shape)

                # ----- 중심점 계산 -----
                center = marker_corners.mean(axis=0)
                cx, cy = int(center[0]), int(center[1])

                # ----- 각도 및 면적 계산 -----
                angle = self._calculate_angle(marker_corners)

                # ----- 결과 객체 생성 -----
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
                    self.logger.debug(f"마커 검증 실패: {msg}")

                candidates.append(detection)

        # ========== 최선의 후보 선택 ==========
        if not candidates:
            return None

        # 우선순위: 1) 검증 통과, 2) 면적이 큰 것 (가까운 것)
        valid_candidates = [c for c in candidates if c.is_valid]

        if valid_candidates:
            # 검증 통과한 것 중 가장 큰 것
            best = max(valid_candidates, key=lambda c: c.area)
            self.logger.debug(f"✓ 최선의 마커 선택: ID={best.marker_id}, 면적={best.area:.0f}px")
            return best
        else:
            # 검증 통과한 것이 없으면 가장 큰 것이라도 반환 (경고 포함)
            best = max(candidates, key=lambda c: c.area)
            self.logger.warning(f"⚠ 검증 실패했지만 마커 반환: {best.validation_message}")
            return best


    def _validate_marker(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, ...]
    ) -> Tuple[bool, str]:
        """
        마커 유효성 검증 (관대한 기준)

        목적: ArUco는 "어디를 볼지"만 알려주는 ROI 마커
        → 검증은 최소한만, ZED Depth가 실제 정확도를 보장

        검증 항목:
        1. 이미지 경계에 너무 가깝지 않은지 (최소한의 체크)
        2. 마커가 너무 작지 않은지 (노이즈 제거용)

        Args:
            corners: 마커 4개 코너 좌표 (shape: 4x2)
            image_shape: 이미지 shape (H, W, ...)

        Returns:
            Tuple[bool, str]: (검증 성공 여부, 메시지)
        """
        h, w = image_shape[:2]

        # ----- 경계 체크 (관대하게) -----
        # 마커 코너가 이미지 밖으로 나가지만 않으면 OK
        margin = 5  # 최소한의 여유만
        if (np.any(corners < margin) or
            np.any(corners[:, 0] > w - margin) or
            np.any(corners[:, 1] > h - margin)):
            return False, "마커가 이미지 경계 밖"

        # ----- 크기 체크 (노이즈 제거용만) -----
        area = cv2.contourArea(corners.astype(np.float32))

        # 최소 면적: 400px (약 20x20) - 옛날 코드 기준
        # 이는 노이즈나 오검출 제거용이지, "멀어서 부정확하다"는 의미 아님
        if area < 400:
            return False, f"마커가 너무 작음 (면적: {area:.0f}px, 노이즈 가능성)"

        # 최대 면적은 체크 안 함 (가까워도 ZED depth는 정확)
        # 굳이 체크한다면 전체 이미지의 90% 이상일 때만 (극단적인 경우)
        if area > (w * h * 0.9):
            return False, f"마커가 화면을 거의 다 차지함 (면적: {area:.0f}px)"

        return True, "정상"

    def _calculate_angle(self, corners: np.ndarray) -> float:
        """
        마커 회전 각도 계산 (Roll 각도)

        **주의**: 이 값은 정보 표시용일 뿐, reject 기준이 아님!
        - 마커가 기울어져도 ZED Depth는 정확함
        - Pose estimation 없이는 Pitch/Yaw 구분 불가
        - 단순 참고용 지표

        마커 상단 엣지의 회전 각도를 계산.
        0도 = 수평, 90도 = 수직 (시계방향)

        Args:
            corners: 마커 4개 코너 좌표 (좌상 → 우상 → 우하 → 좌하)

        Returns:
            float: 회전 각도 (도 단위, -180 ~ 180)
        """
        # 코너 순서: 좌상(0) → 우상(1)
        v = corners[1] - corners[0]  # 상단 가로 벡터

        # atan2를 사용하여 벡터의 각도 계산
        angle_rad = np.arctan2(v[1], v[0])
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def draw_marker(
        self,
        image: np.ndarray,
        detection: MarkerDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> None:
        """
        이미지에 마커 시각화 그리기

        그리는 요소:
        1. 마커 영역 반투명 채우기
        2. 마커 외곽선
        3. 중심점
        4. 마커 ID 텍스트

        Args:
            image: 그릴 이미지 (in-place 수정됨)
            detection: 마커 감지 결과
            color: 그리기 색상 (BGR)
            thickness: 외곽선 두께
        """
        # ===== 1. 마커 영역 반투명 채우기 =====
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        # 원본과 블렌딩 (alpha=0.3으로 반투명 효과)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # ===== 2. 마커 외곽선 =====
        cv2.polylines(image, [pts], True, color, thickness)

        # ===== 3. 중심점 =====
        cv2.circle(
            image,
            (detection.center_x, detection.center_y),
            8,      # 반지름
            color,
            -1      # 채움
        )

        # ===== 4. 마커 ID 텍스트 =====
        # 그림자 효과 (검은색 배경 + 컬러 텍스트)
        text_pos = (detection.center_x + 20, detection.center_y - 10)
        cv2.putText(
            image,
            f"ID: {detection.marker_id}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # 검은색 그림자
            4
        )
        cv2.putText(
            image,
            f"ID: {detection.marker_id}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,      # 지정된 색상
            2
        )
