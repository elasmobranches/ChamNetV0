"""
Configuration management module
YAML 기반 설정을 로드하고 관리합니다.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import os


class ConfigError(Exception):
    """설정 관련 에러"""
    pass


class Config:
    """애플리케이션 설정 관리 클래스"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        """
        if config_path is None:
            # 환경 변수에서 설정 경로를 가져오거나 기본값 사용
            base_dir = Path(__file__).parent.parent.parent
            config_path = os.getenv(
                'DEPTH_CONFIG_PATH',
                str(base_dir / 'config' / 'config.yaml')
            )

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """설정 파일을 로드합니다."""
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
        """필수 설정 항목을 검증합니다."""
        required_keys = ['marker_id', 'marker_size_mm', 'aruco_dict', 'zed_settings']
        missing_keys = [key for key in required_keys if key not in self._config]

        if missing_keys:
            raise ConfigError(f"필수 설정 항목이 누락되었습니다: {missing_keys}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값을 가져옵니다.

        Args:
            key: 설정 키 (점 표기법 지원, 예: 'zed_settings.fps')
            default: 기본값

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

    # Marker 설정
    @property
    def marker_id(self) -> int:
        return self._config['marker_id']

    @property
    def marker_size_mm(self) -> float:
        return self._config['marker_size_mm']

    @property
    def aruco_dict(self) -> str:
        return self._config['aruco_dict']

    @property
    def target_distances(self) -> List[float]:
        return self._config.get('target_distances', [])

    @property
    def measurements_per_distance(self) -> int:
        return self._config.get('measurements_per_distance', 50)

    # ZED 설정
    @property
    def zed_settings(self) -> Dict[str, Any]:
        return self._config.get('zed_settings', {})

    @property
    def depth_mode(self) -> str:
        return self.zed_settings.get('depth_mode', 'NEURAL_PLUS')

    @property
    def resolution(self) -> str:
        return self.zed_settings.get('resolution', 'HD1080')

    @property
    def fps(self) -> int:
        return self.zed_settings.get('fps', 30)

    @property
    def depth_min_distance(self) -> int:
        return self.zed_settings.get('depth_minimum_distance', 200)

    @property
    def depth_max_distance(self) -> int:
        return self.zed_settings.get('depth_maximum_distance', 20000)

    @property
    def confidence_threshold(self) -> int:
        return self.zed_settings.get('confidence_threshold', 0)

    @property
    def texture_confidence_threshold(self) -> int:
        return self.zed_settings.get('texture_confidence_threshold', 0)

    @property
    def enable_fill_mode(self) -> bool:
        return self.zed_settings.get('enable_fill_mode', True)

    @property
    def depth_stabilization(self) -> int:
        return self.zed_settings.get('depth_stabilization', 1)

    # Depth 측정 설정
    @property
    def depth_measurement(self) -> Dict[str, Any]:
        return self._config.get('depth_measurement', {})

    @property
    def use_marker_region(self) -> bool:
        return self.depth_measurement.get('use_marker_region', True)

    @use_marker_region.setter
    def use_marker_region(self, value: bool) -> None:
        """측정 모드 설정 (True: 마커 영역, False: 윈도우)"""
        if 'depth_measurement' not in self._config:
            self._config['depth_measurement'] = {}
        self._config['depth_measurement']['use_marker_region'] = value

    @property
    def window_size(self) -> int:
        return self.depth_measurement.get('window_size', 11)

    @property
    def min_valid_ratio(self) -> float:
        return self.depth_measurement.get('min_valid_ratio', 0.5)

    # 웹 서버 설정
    @property
    def web_host(self) -> str:
        return os.getenv('WEB_HOST', self.get('web_server.host', '0.0.0.0'))

    @property
    def web_port(self) -> int:
        return int(os.getenv('WEB_PORT', self.get('web_server.port', 5000)))

    @property
    def stream_quality(self) -> int:
        return self.get('web_server.stream_quality', 85)

    @property
    def stream_fps(self) -> int:
        return self.get('web_server.stream_fps', 15)

    # 파일 경로 설정
    @property
    def output_dir(self) -> Path:
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / self.get('paths.output_dir', 'data/results')

    @property
    def log_dir(self) -> Path:
        base_dir = Path(__file__).parent.parent.parent
        return base_dir / self.get('paths.log_dir', 'logs')

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"
