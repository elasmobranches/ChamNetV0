"""
ZED Camera management module
ZED 2i 카메라 초기화 및 프레임 캡처를 담당합니다.
"""

import numpy as np
import pyzed.sl as sl
from typing import Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import Config


class ZEDCameraError(Exception):
    """ZED 카메라 관련 에러"""
    pass


class ZEDCamera:
    """ZED 카메라 관리 클래스"""

    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = get_logger('depth_estimation.camera')
        self.camera: Optional[sl.Camera] = None
        self._is_opened = False

        # 이미지 및 depth 매트릭스
        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()

    def open(self) -> None:
        """
        ZED 카메라를 초기화하고 엽니다.

        Raises:
            ZEDCameraError: 카메라 초기화 실패 시
        """
        try:
            self.camera = sl.Camera()

            # 초기화 파라미터 설정
            init_params = sl.InitParameters()

            # Depth 모드 설정
            depth_mode_map = {
                'NEURAL_PLUS': sl.DEPTH_MODE.NEURAL_PLUS,
                'NEURAL': sl.DEPTH_MODE.NEURAL,
                'ULTRA': sl.DEPTH_MODE.ULTRA,
                'QUALITY': sl.DEPTH_MODE.QUALITY,
                'PERFORMANCE': sl.DEPTH_MODE.PERFORMANCE
            }
            init_params.depth_mode = depth_mode_map.get(
                self.config.depth_mode,
                sl.DEPTH_MODE.NEURAL_PLUS
            )

            # 해상도 설정
            resolution_map = {
                'HD2K': sl.RESOLUTION.HD2K,
                'HD1080': sl.RESOLUTION.HD1080,
                'HD720': sl.RESOLUTION.HD720,
                'VGA': sl.RESOLUTION.VGA
            }
            init_params.camera_resolution = resolution_map.get(
                self.config.resolution,
                sl.RESOLUTION.HD1080
            )

            # 기타 파라미터
            init_params.depth_stabilization = self.config.depth_stabilization
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_minimum_distance = self.config.depth_min_distance
            init_params.depth_maximum_distance = self.config.depth_max_distance
            init_params.camera_fps = self.config.fps

            # 카메라 열기
            err = self.camera.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ZEDCameraError(f"카메라 열기 실패: {err}")

            # 카메라 설정
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
            self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)

            self._is_opened = True

            # 카메라 정보 로그
            cam_info = self.camera.get_camera_information()
            self.logger.info(f"ZED 카메라 초기화 완료")
            self.logger.info(f"  - 해상도: {self.config.resolution}")
            self.logger.info(f"  - FPS: {self.config.fps}")
            self.logger.info(f"  - Depth 모드: {self.config.depth_mode}")

        except Exception as e:
            self.logger.error(f"ZED 카메라 초기화 실패: {e}")
            raise ZEDCameraError(f"카메라 초기화 실패: {e}")

    def grab_frame(self) -> bool:
        """
        새 프레임을 캡처합니다.

        Returns:
            성공 여부

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        runtime_params = self._get_runtime_params()
        return self.camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS

    def get_image(self) -> np.ndarray:
        """
        현재 프레임의 이미지를 가져옵니다.

        Returns:
            RGB 이미지 (numpy array)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        self.camera.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        image = self._image_mat.get_data()[:, :, :3].copy()
        return image

    def get_depth(self) -> np.ndarray:
        """
        현재 프레임의 depth 맵을 가져옵니다.

        Returns:
            Depth 맵 (numpy array, 단위: mm)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        self.camera.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        depth = self._depth_mat.get_data().copy()
        return depth

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 프레임의 이미지와 depth 맵을 모두 가져옵니다.

        Returns:
            (이미지, depth 맵) 튜플

        Raises:
            ZEDCameraError: 프레임 가져오기 실패 시
        """
        if not self.grab_frame():
            raise ZEDCameraError("프레임 캡처 실패")

        return self.get_image(), self.get_depth()

    def _get_runtime_params(self) -> sl.RuntimeParameters:
        """런타임 파라미터를 생성합니다."""
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = self.config.confidence_threshold
        runtime.texture_confidence_threshold = self.config.texture_confidence_threshold
        runtime.enable_depth = True
        runtime.enable_fill_mode = self.config.enable_fill_mode
        return runtime

    def close(self) -> None:
        """카메라를 닫습니다."""
        if self._is_opened and self.camera:
            self.camera.close()
            self._is_opened = False
            self.logger.info("ZED 카메라 종료")

    def is_opened(self) -> bool:
        """카메라가 열려있는지 확인합니다."""
        return self._is_opened

    def __enter__(self):
        """Context manager 진입"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()

    def __del__(self):
        """소멸자"""
        self.close()
