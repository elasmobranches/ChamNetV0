"""
ZED Camera Management Module
============================
ZED 2i 스테레오 카메라 제어를 담당하는 모듈

ZED SDK(pyzed)를 래핑하여 간편한 인터페이스 제공:
- 카메라 초기화 및 설정
- RGB 이미지 + Depth 맵 동시 캡처
- 런타임 파라미터 관리

ZED SDK 문서: https://www.stereolabs.com/docs/
"""

import numpy as np
import pyzed.sl as sl
from typing import Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import Config


class ZEDCameraError(Exception):
    """
    ZED 카메라 관련 예외

    발생 상황:
    - 카메라 초기화 실패 (USB 연결 문제, SDK 버전 불일치 등)
    - 프레임 캡처 실패 (카메라 연결 끊김 등)
    """
    pass


class ZEDCamera:
    """
    ZED 카메라 관리 클래스

    ZED SDK의 복잡한 API를 래핑하여 단순한 인터페이스 제공.
    RGB 이미지와 Depth 맵을 동시에 가져올 수 있음.

    사용 예시:
        camera = ZEDCamera(config)
        camera.open()
        image, depth = camera.get_frame()
        camera.close()

    또는 Context Manager 사용:
        with ZEDCamera(config) as camera:
            image, depth = camera.get_frame()

    Attributes:
        config: 설정 객체
        camera: ZED SDK Camera 인스턴스
    """

    def __init__(self, config: Config):
        """
        ZED 카메라 초기화

        Args:
            config: 설정 객체 (해상도, FPS, Depth 모드 등 포함)
        """
        self.config = config
        self.logger = get_logger('depth_estimation.camera')

        # ZED SDK 카메라 객체 (open() 호출 전까지 None)
        self.camera: Optional[sl.Camera] = None
        self._is_opened = False

        # 이미지/Depth 저장용 매트릭스 (재사용하여 메모리 할당 최소화)
        self._image_mat = sl.Mat()  # RGBA 이미지
        self._depth_mat = sl.Mat()  # Depth 맵 (mm 단위)
        self._confidence_mat = sl.Mat()  # Confidence 맵 (0-100)

    def open(self) -> None:
        """
        ZED 카메라를 열고 초기화

        이 메서드는 ZED 카메라를 시스템에 연결하고, config.yaml의 설정을
        ZED SDK 파라미터로 변환하여 적용합니다.

        초기화 단계:
            1. InitParameters 생성 (Depth 모드, 해상도, FPS 등)
            2. camera.open() 호출 → USB 연결 및 펌웨어 로드
            3. 최적 카메라 설정 적용 (line 136 위치)
            4. 카메라 정보 로깅

        중요:
            - line 136에 debug_exposure.py로 찾은 최적 설정을 추가하세요
            - Auto Exposure가 부적절할 경우 Manual 모드로 전환 권장
            - 조명 환경이 바뀌면 재설정 필요

        Raises:
            ZEDCameraError: 카메라 열기 또는 초기화 실패 시
                - USB 연결 문제
                - ZED SDK 버전 불일치
                - 카메라 펌웨어 오류
                - 다른 프로세스가 카메라 사용 중
        """
        try:
            self.camera = sl.Camera()

            # ========== 초기화 파라미터 설정 ==========
            init_params = sl.InitParameters()

            # ----- Depth 모드 설정 -----
            # NEURAL_PLUS: AI 기반 (가장 정확, GPU 필요)
            # ULTRA: 고품질 스테레오 매칭
            # QUALITY: 품질 우선
            # PERFORMANCE: 속도 우선
            depth_mode_map = {
                'NEURAL_PLUS': sl.DEPTH_MODE.NEURAL_PLUS,
                'NEURAL': sl.DEPTH_MODE.NEURAL,
                'ULTRA': sl.DEPTH_MODE.ULTRA,
                'QUALITY': sl.DEPTH_MODE.QUALITY,
                'PERFORMANCE': sl.DEPTH_MODE.PERFORMANCE
            }
            init_params.depth_mode = depth_mode_map.get(
                self.config.depth_mode,
                sl.DEPTH_MODE.NEURAL_PLUS  # 기본값
            )

            # ----- 해상도 설정 -----
            # HD2K: 2208x1242 (고해상도, 느림)
            # HD1080: 1920x1080 (균형)
            # HD720: 1280x720 (빠름)
            # VGA: 672x376 (가장 빠름)
            resolution_map = {
                'HD2K': sl.RESOLUTION.HD2K,
                'HD1080': sl.RESOLUTION.HD1080,
                'HD720': sl.RESOLUTION.HD720,
                'VGA': sl.RESOLUTION.VGA
            }
            init_params.camera_resolution = resolution_map.get(
                self.config.resolution,
                sl.RESOLUTION.HD1080  # 기본값
            )

            # ----- 기타 파라미터 -----
            init_params.depth_stabilization = self.config.depth_stabilization  # Depth 시간적 안정화
            init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth 단위: mm
            init_params.depth_minimum_distance = self.config.depth_min_distance  # 최소 거리 (mm)
            init_params.depth_maximum_distance = self.config.depth_max_distance  # 최대 거리 (mm)
            init_params.camera_fps = self.config.fps  # 카메라 FPS

            # ========== 카메라 열기 ==========
            err = self.camera.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ZEDCameraError(f"카메라 열기 실패: {err}")

            self._is_opened = True

            # ========== 최적 카메라 설정 적용 (여기에 추가!) ==========
            # debug_exposure.py로 찾은 최적값을 여기에 하드코딩하세요.
            #
            # 예시 (debug_exposure.py 실행 후 "설정 저장" 버튼 클릭 시 터미널 출력):
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)        # Auto Exposure 끄기
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 65)     # 노출값
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 45)         # 게인값
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)    # 밝기
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)      # 대비
            #   self.camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)     # 선명도
            #   self.logger.info("[카메라 설정] 최적값 적용 완료")
            #
            # 주의사항:
            #   - 위 코드는 예시입니다. 실제 환경에서 찾은 값을 사용하세요!
            #   - 조명 환경이 바뀌면 다시 debug_exposure.py 실행 필요
            #   - 마커 검출률 90% 이상 달성 시의 값을 사용
            #   - 설정 날짜를 주석에 기록하면 관리가 편리함
            # ==============================================================

            # [DEBUG] 현재 노출/게인 값 로깅
            try:
                settings = self.get_exposure_settings()
                self.logger.info(f"[노출 설정] Auto Exposure/Gain 활성화")
                self.logger.info(f"  - 현재 EXPOSURE: {settings['exposure']}")
                self.logger.info(f"  - 현재 GAIN: {settings['gain']}")
            except Exception as e:
                self.logger.warning(f"노출 설정 로깅 실패: {e}")

            # ========== 초기화 완료 로그 ==========
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
        새 프레임 캡처 (내부 버퍼에 저장)

        grab() 호출 후 get_image(), get_depth()로 데이터 조회 가능.
        보통은 get_frame()을 직접 사용하는 것이 편리함.

        Returns:
            bool: 캡처 성공 여부

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        runtime_params = self._get_runtime_params()
        return self.camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS

    def get_image(self) -> np.ndarray:
        """
        현재 프레임의 RGB 이미지 조회

        grab_frame() 또는 get_frame() 호출 후 사용.
        ZED는 RGBA를 반환하므로 RGB만 추출하여 반환.

        Returns:
            np.ndarray: RGB 이미지 (shape: H x W x 3, dtype: uint8)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # 왼쪽 카메라 이미지 조회 (ZED는 스테레오이므로 LEFT/RIGHT 선택 가능)
        self.camera.retrieve_image(self._image_mat, sl.VIEW.LEFT)

        # RGBA → RGB 변환 (4채널 → 3채널)
        image = self._image_mat.get_data()[:, :, :3].copy()
        return image

    def get_depth(self) -> np.ndarray:
        """
        현재 프레임의 Depth 맵 조회

        grab_frame() 또는 get_frame() 호출 후 사용.
        Depth 값은 밀리미터(mm) 단위.

        Returns:
            np.ndarray: Depth 맵 (shape: H x W, dtype: float32, 단위: mm)
                       유효하지 않은 픽셀은 inf 또는 nan

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # Depth 맵 조회 (MEASURE.DEPTH: 각 픽셀의 Z 거리)
        self.camera.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        depth = self._depth_mat.get_data().copy()
        return depth

    def get_confidence(self) -> np.ndarray:
        """
        현재 프레임의 Confidence 맵 조회

        grab_frame() 또는 get_frame() 호출 후 사용.
        Confidence 값은 0~100 범위 (100이 가장 신뢰도가 낮음)

        Returns:
            np.ndarray: Confidence 맵 (shape: H x W, dtype: float32, 범위: 0-100)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # Confidence 맵 조회 (MEASURE.CONFIDENCE: 각 픽셀의 depth 신뢰도)
        self.camera.retrieve_measure(self._confidence_mat, sl.MEASURE.CONFIDENCE)
        confidence = self._confidence_mat.get_data().copy()
        return confidence

    def get_frame(self, debug_exposure: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RGB 이미지, Depth 맵, Confidence 맵을 동시에 가져오기

        가장 일반적으로 사용하는 메서드.
        내부적으로 grab_frame() → get_image() + get_depth() + get_confidence() 순서로 호출.

        Args:
            debug_exposure: True면 프레임마다 노출/게인 값을 로깅

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (RGB 이미지, Depth 맵, Confidence 맵)

        Raises:
            ZEDCameraError: 프레임 캡처 실패 시
        """
        if not self.grab_frame():
            raise ZEDCameraError("프레임 캡처 실패")

        # [DEBUG] 노출 값 로깅
        if debug_exposure:
            settings = self.get_exposure_settings()
            self.logger.debug(f"[프레임 노출] EXPOSURE={settings['exposure']}, GAIN={settings['gain']}")

        return self.get_image(), self.get_depth(), self.get_confidence()

    def set_manual_exposure(self, exposure: int, gain: int) -> None:
        """
        수동 노출/게인 설정 (Auto Exposure 끄기)

        Args:
            exposure: 노출 값 (0-100, 환경에 맞게 조절)
            gain: 게인 값 (0-100, 환경에 맞게 조절)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # Auto Exposure/Gain 끄기
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)

        # 수동 값 설정
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

        # 설정된 값 확인
        settings = self.get_exposure_settings()

        self.logger.info(f"[노출 고정] Auto Exposure/Gain 비활성화")
        self.logger.info(f"  - EXPOSURE 설정: {exposure} → 실제: {settings['exposure']}")
        self.logger.info(f"  - GAIN 설정: {gain} → 실제: {settings['gain']}")

    def set_auto_exposure(self) -> None:
        """
        자동 노출/게인 설정 (Auto Exposure 켜기)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # Auto Exposure/Gain 켜기
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)

        settings = self.get_exposure_settings()

        self.logger.info(f"[노출 자동] Auto Exposure/Gain 활성화")
        self.logger.info(f"  - 현재 EXPOSURE: {settings['exposure']}")
        self.logger.info(f"  - 현재 GAIN: {settings['gain']}")

    def get_exposure_settings(self) -> dict:
        """
        현재 노출/게인 설정 조회

        Returns:
            dict: {'exposure': int, 'gain': int, 'aec_agc': int}

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        # ZED SDK의 get_camera_settings는 int 또는 (err, int) 튜플을 반환
        try:
            exposure_val = self.camera.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
            gain_val = self.camera.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
            aec_agc_val = self.camera.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC)

            # 튜플인 경우 두 번째 값 추출
            if isinstance(exposure_val, tuple):
                exposure_val = exposure_val[1] if len(exposure_val) > 1 else exposure_val[0]
            if isinstance(gain_val, tuple):
                gain_val = gain_val[1] if len(gain_val) > 1 else gain_val[0]
            if isinstance(aec_agc_val, tuple):
                aec_agc_val = aec_agc_val[1] if len(aec_agc_val) > 1 else aec_agc_val[0]

            # ERROR_CODE 타입인 경우 int로 변환 (value 속성 사용)
            if hasattr(exposure_val, 'value'):
                exposure_val = exposure_val.value
            if hasattr(gain_val, 'value'):
                gain_val = gain_val.value
            if hasattr(aec_agc_val, 'value'):
                aec_agc_val = aec_agc_val.value

            return {
                'exposure': int(exposure_val),
                'gain': int(gain_val),
                'aec_agc': int(aec_agc_val)
            }
        except Exception as e:
            self.logger.error(f"노출 설정 조회 실패: {e}")
            return {'exposure': 0, 'gain': 0, 'aec_agc': 0}

    def set_brightness(self, value: int) -> None:
        """
        밝기 설정

        Args:
            value: 밝기 값 (0-8, 기본값: 4)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        value = max(0, min(8, value))  # 0-8 범위로 제한
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, value)
        self.logger.info(f"밝기 설정: {value}")

    def set_contrast(self, value: int) -> None:
        """
        대비 설정

        Args:
            value: 대비 값 (0-8, 기본값: 4)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        value = max(0, min(8, value))  # 0-8 범위로 제한
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, value)
        self.logger.info(f"대비 설정: {value}")

    def set_sharpness(self, value: int) -> None:
        """
        선명도 설정

        Args:
            value: 선명도 값 (0-8, 기본값: 4)

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        value = max(0, min(8, value))  # 0-8 범위로 제한
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, value)
        self.logger.info(f"선명도 설정: {value}")

    def get_camera_settings(self) -> dict:
        """
        모든 카메라 설정 조회

        Returns:
            dict: 모든 VIDEO_SETTINGS 값
                {
                    'exposure': int,
                    'gain': int,
                    'aec_agc': int,
                    'brightness': int,
                    'contrast': int,
                    'sharpness': int
                }

        Raises:
            ZEDCameraError: 카메라가 열려있지 않은 경우
        """
        if not self._is_opened:
            raise ZEDCameraError("카메라가 열려있지 않습니다")

        def get_setting(setting):
            """설정 값 추출 헬퍼"""
            try:
                val = self.camera.get_camera_settings(setting)
                # 튜플인 경우 두 번째 값 추출
                if isinstance(val, tuple):
                    val = val[1] if len(val) > 1 else val[0]
                # ERROR_CODE 타입인 경우 int로 변환
                if hasattr(val, 'value'):
                    val = val.value
                return int(val)
            except Exception:
                return 0

        return {
            'exposure': get_setting(sl.VIDEO_SETTINGS.EXPOSURE),
            'gain': get_setting(sl.VIDEO_SETTINGS.GAIN),
            'aec_agc': get_setting(sl.VIDEO_SETTINGS.AEC_AGC),
            'brightness': get_setting(sl.VIDEO_SETTINGS.BRIGHTNESS),
            'contrast': get_setting(sl.VIDEO_SETTINGS.CONTRAST),
            'sharpness': get_setting(sl.VIDEO_SETTINGS.SHARPNESS)
        }

    def _get_runtime_params(self) -> sl.RuntimeParameters:
        """
        런타임 파라미터 생성

        grab() 호출 시마다 전달되는 파라미터.
        Depth 계산 관련 옵션을 제어함.

        Returns:
            sl.RuntimeParameters: 런타임 파라미터 객체
        """
        runtime = sl.RuntimeParameters()

        # 낮을수록 엄격하게 필터링 (신뢰도 높은 픽셀만 포함, 노이즈 감소, 유효 데이터 줄어듦)
        # 높을수록 관대하게 필터링 (더 많은 픽셀 포함, 노이즈 증가)
        runtime.confidence_threshold = self.config.confidence_threshold

        # Texture Confidence: 텍스처가 부족한 영역 필터링
        runtime.texture_confidence_threshold = self.config.texture_confidence_threshold

        # Depth 계산 활성화
        runtime.enable_depth = True

        # Fill Mode: Depth 홀(구멍)을 주변 값으로 채움
        runtime.enable_fill_mode = self.config.enable_fill_mode

        return runtime

    def close(self) -> None:
        """
        카메라 닫기 및 리소스 해제

        프로그램 종료 전 반드시 호출해야 함.
        Context Manager 사용 시 자동 호출됨.
        """
        if self._is_opened and self.camera:
            self.camera.close()
            self._is_opened = False
            self.logger.info("ZED 카메라 종료")

    def is_opened(self) -> bool:
        """
        카메라가 열려있는지 확인

        Returns:
            bool: 열려있으면 True
        """
        return self._is_opened

    # ========== Context Manager 지원 ==========
    def __enter__(self):
        """Context Manager 진입: 카메라 열기"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 종료: 카메라 닫기"""
        self.close()

    def __del__(self):
        """소멸자: 리소스 정리"""
        self.close()
