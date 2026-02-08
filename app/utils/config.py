"""
Configuration Management Module
================================
YAML 기반 설정을 로드하고 관리하는 모듈

설정 파일 구조 (config/config.yaml):
- marker_id: ArUco 마커 ID
- marker_size_mm: 마커 실제 크기 (mm)
- aruco_dict: ArUco 딕셔너리 종류
- zed_settings: ZED 카메라 설정
- depth_measurement: Depth 측정 설정
- web_server: Flask 웹 서버 설정
- paths: 출력 경로 설정

환경 변수 지원:
- DEPTH_CONFIG_PATH: 설정 파일 경로 오버라이드
- WEB_HOST: 웹 서버 호스트 오버라이드
- WEB_PORT: 웹 서버 포트 오버라이드
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import os


class ConfigError(Exception):
    """
    설정 관련 예외

    발생 상황:
    - 설정 파일을 찾을 수 없음
    - YAML 파싱 오류
    - 필수 설정 항목 누락
    """
    pass


class Config:
    """
    애플리케이션 설정 관리 클래스

    YAML 설정 파일을 로드하고, property로 타입 안전한 접근 제공.
    점 표기법(dot notation)으로 중첩된 설정값 조회 가능.

    사용 예시:
        config = Config('config/config.yaml')
        print(config.marker_id)           # 직접 접근
        print(config.depth_mode)          # 중첩 설정 (자동 추출)
        print(config.get('zed_settings.fps'))  # 점 표기법

    Attributes:
        config_path: 설정 파일 경로
        _config: 로드된 설정 딕셔너리
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        설정 관리자 초기화

        Args:
            config_path: 설정 파일 경로
                        None이면 환경변수 또는 기본 경로 사용
        """
        if config_path is None:
            # 환경 변수에서 설정 경로를 가져오거나 기본값 사용
            # 기본 경로: {프로젝트루트}/config/config.yaml
            base_dir = Path(__file__).parent.parent.parent
            config_path = os.getenv(
                'DEPTH_CONFIG_PATH',
                str(base_dir / 'config' / 'config.yaml')
            )

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        설정 파일 로드

        YAML 파일을 파싱하고 필수 키를 검증.

        Raises:
            ConfigError: 파일이 없거나 파싱 오류, 필수 키 누락
        """
        try:
            if not self.config_path.exists():
                raise ConfigError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            # 필수 키 검증
            self._validate()

        except yaml.YAMLError as e:
            raise ConfigError(f"설정 파일 파싱 오류: {e}")
        except Exception as e:
            raise ConfigError(f"설정 로드 실패: {e}")

    def _validate(self) -> None:
        """
        필수 설정 항목 검증

        Raises:
            ConfigError: 필수 키가 누락된 경우
        """
        required_keys = ['marker_id', 'marker_size_mm', 'aruco_dict', 'zed_settings']
        missing_keys = [key for key in required_keys if key not in self._config]

        if missing_keys:
            raise ConfigError(f"필수 설정 항목이 누락되었습니다: {missing_keys}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 조회 (점 표기법 지원)

        예시:
            config.get('marker_id')           # 'marker_id' 키
            config.get('zed_settings.fps')    # 'zed_settings' 내의 'fps'
            config.get('missing_key', 30)     # 없으면 기본값 30 반환

        Args:
            key: 설정 키 (점으로 중첩 키 구분)
            default: 키가 없을 때 반환할 기본값

        Returns:
            설정 값 또는 기본값
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # ==================== Marker 설정 ====================
    @property
    def marker_id(self) -> int:
        """ArUco 마커 ID (0~49 for DICT_4X4_50)"""
        return self._config['marker_id']

    @property
    def marker_size_mm(self) -> int:
        """마커 실제 크기 (mm) - 정확히 측정 필요!"""
        return self._config['marker_size_mm']

    @marker_size_mm.setter
    def marker_size_mm(self, value: int) -> None:
        """
        마커 크기 설정 (런타임 변경 가능)

        Args:
            value: 마커 크기 (mm) - 50, 100, 250 등
        """
        self._config['marker_size_mm'] = value

    @property
    def aruco_dict(self) -> str:
        """ArUco 딕셔너리 이름 (예: 'DICT_4X4_50')"""
        return self._config['aruco_dict']

    @property
    def target_distances(self) -> List[float]:
        """측정할 목표 거리 리스트 (m)"""
        return self._config.get('target_distances', [])

    @property
    def measurements_per_distance(self) -> int:
        """거리당 측정 횟수"""
        return self._config.get('measurements_per_distance', 50)

    # ==================== ZED 카메라 설정 ====================
    @property
    def zed_settings(self) -> Dict[str, Any]:
        """ZED 카메라 설정 딕셔너리"""
        return self._config.get('zed_settings', {})

    @property
    def depth_mode(self) -> str:
        """
        Depth 모드
        - NEURAL_PLUS: AI 기반 (가장 정확)
        - ULTRA: 고품질 스테레오 매칭
        - QUALITY: 품질 우선
        - PERFORMANCE: 속도 우선
        """
        return self.zed_settings.get('depth_mode', 'NEURAL_PLUS')

    @property
    def resolution(self) -> str:
        """
        카메라 해상도
        - HD2K: 2208x1242
        - HD1080: 1920x1080
        - HD720: 1280x720
        - VGA: 672x376
        """
        return self.zed_settings.get('resolution', 'HD1080')

    def get_resolution_dimensions(self) -> tuple:
        """
        현재 설정된 해상도의 실제 크기 반환

        Returns:
            (width, height) 튜플

        Example:
            width, height = config.get_resolution_dimensions()
            # HD1080 → (1920, 1080)
        """
        resolution_map = {
            'HD2K': (2208, 1242),
            'HD1080': (1920, 1080),
            'HD720': (1280, 720),
            'VGA': (672, 376)
        }
        return resolution_map.get(self.resolution, (1920, 1080))  # 기본값: HD1080

    @property
    def fps(self) -> int:
        """카메라 FPS (15, 30, 60 등)"""
        return self.zed_settings.get('fps', 30)

    @property
    def depth_min_distance(self) -> int:
        """최소 Depth 거리 (mm) - 이보다 가까운 건 무시"""
        return self.zed_settings.get('depth_minimum_distance', 200)

    @property
    def depth_max_distance(self) -> int:
        """최대 Depth 거리 (mm) - 이보다 먼 건 무시"""
        return self.zed_settings.get('depth_maximum_distance', 20000)

    @property
    def confidence_threshold(self) -> int:
        """
        Confidence 임계값 (0~100)
        - 0: 모든 픽셀 포함 (노이즈 많음)
        - 100: 확실한 픽셀만 (구멍 많음)
        """
        return self.zed_settings.get('confidence_threshold', 0)

    @property
    def texture_confidence_threshold(self) -> int:
        """
        Texture Confidence 임계값 (0~100)
        텍스처가 부족한 영역 필터링
        """
        return self.zed_settings.get('texture_confidence_threshold', 0)

    @property
    def enable_fill_mode(self) -> bool:
        """
        Fill Mode 활성화 여부
        True: Depth 구멍을 주변 값으로 채움
        """
        return self.zed_settings.get('enable_fill_mode', True)

    @property
    def depth_stabilization(self) -> int:
        """
        Depth 시간적 안정화 (0~100)
        높을수록 시간에 따른 변동 감소, 반응 느려짐
        """
        return self.zed_settings.get('depth_stabilization', 1)

    # ==================== Depth 측정 설정 ====================
    @property
    def depth_measurement(self) -> Dict[str, Any]:
        """Depth 측정 설정 딕셔너리"""
        return self._config.get('depth_measurement', {})

    @property
    def use_marker_region(self) -> bool:
        """
        마커 영역 모드 사용 여부
        - True: 마커 내부 전체 영역 (권장, 정확)
        - False: 마커 중심 주변 윈도우만 (빠름)
        """
        return self.depth_measurement.get('use_marker_region', True)

    @use_marker_region.setter
    def use_marker_region(self, value: bool) -> None:
        """
        측정 모드 설정 (런타임 변경 가능)

        Args:
            value: True=마커 영역, False=윈도우
        """
        if 'depth_measurement' not in self._config:
            self._config['depth_measurement'] = {}
        self._config['depth_measurement']['use_marker_region'] = value

    @property
    def window_size(self) -> int:
        """윈도우 모드 시 윈도우 크기 (픽셀, 홀수 권장)"""
        return self.depth_measurement.get('window_size', 11)

    @property
    def min_valid_ratio(self) -> float:
        """최소 유효 픽셀 비율 (0~1) - 이보다 낮으면 측정 무효"""
        return self.depth_measurement.get('min_valid_ratio', 0.5)

    # ==================== 웹 서버 설정 ====================
    @property
    def web_host(self) -> str:
        """
        웹 서버 호스트
        - '0.0.0.0': 모든 네트워크 인터페이스
        - '127.0.0.1': 로컬만
        환경변수 WEB_HOST로 오버라이드 가능
        """
        return os.getenv('WEB_HOST', self.get('web_server.host', '0.0.0.0'))

    @property
    def web_port(self) -> int:
        """
        웹 서버 포트
        환경변수 WEB_PORT로 오버라이드 가능
        """
        return int(os.getenv('WEB_PORT', self.get('web_server.port', 5000)))

    @property
    def stream_quality(self) -> int:
        """JPEG 스트림 품질 (1~100) - 높을수록 선명, 느림"""
        return self.get('web_server.stream_quality', 85)

    @property
    def stream_fps(self) -> int:
        """웹 스트리밍 FPS - 낮을수록 네트워크 부하 감소"""
        return self.get('web_server.stream_fps', 15)

    # ==================== 파일 경로 설정 ====================
    @property
    def output_dir(self) -> Path:
        """결과 출력 디렉토리 (CSV, 이미지)"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / self.get('paths.output_dir', 'data/results')

    @property
    def log_dir(self) -> Path:
        """로그 파일 디렉토리"""
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / self.get('paths.log_dir', 'logs')

    def __repr__(self) -> str:
        """문자열 표현"""
        return f"Config(path={self.config_path})"
