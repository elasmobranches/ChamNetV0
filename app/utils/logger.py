"""
Logging Configuration Module
============================
애플리케이션 로깅을 설정하고 관리하는 모듈

로그 출력 위치:
1. 콘솔 (터미널) - 컬러 출력 지원
2. 파일 - 일별 로그 파일 생성

로그 레벨:
- DEBUG: 상세한 디버깅 정보
- INFO: 일반 동작 정보 (기본)
- WARNING: 경고 (마커 감지 실패 등)
- ERROR: 오류 (카메라 오류 등)
- CRITICAL: 심각한 오류

로그 파일 위치:
- logs/depth_estimation_YYYYMMDD.log
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    컬러 출력을 지원하는 로그 포매터

    터미널에서 로그 레벨별로 다른 색상으로 출력하여
    가독성을 높임.

    색상:
    - DEBUG: 시안 (하늘색)
    - INFO: 녹색
    - WARNING: 노란색
    - ERROR: 빨간색
    - CRITICAL: 마젠타 (보라색)
    """

    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan (시안)
        'INFO': '\033[32m',       # Green (녹색)
        'WARNING': '\033[33m',    # Yellow (노란색)
        'ERROR': '\033[31m',      # Red (빨간색)
        'CRITICAL': '\033[35m',   # Magenta (마젠타)
    }
    RESET = '\033[0m'  # 색상 리셋

    def format(self, record):
        """
        로그 레코드를 포맷팅 (레벨에 색상 적용)

        Args:
            record: 로그 레코드

        Returns:
            str: 포맷팅된 로그 문자열
        """
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = 'depth_estimation',
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    로거 설정 및 생성

    콘솔과 파일 핸들러를 설정하여 로그를 동시에 출력.
    기존 핸들러는 제거하여 중복 출력 방지.

    Args:
        name: 로거 이름 (모듈별로 다르게 설정 가능)
        log_dir: 로그 파일 저장 디렉토리
        level: 로그 레벨 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: 콘솔 출력 여부
        file_output: 파일 출력 여부

    Returns:
        logging.Logger: 설정된 로거 인스턴스

    사용 예시:
        logger = setup_logger(
            name='depth_estimation',
            log_dir=Path('logs'),
            level=logging.INFO
        )
        logger.info("시작!")
        logger.warning("주의!")
        logger.error("오류 발생!")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    # 같은 로거를 여러 번 setup_logger 호출해도 핸들러가 중복되지 않음
    logger.handlers.clear()

    # ========== 포맷 설정 ==========
    # 콘솔: 간결하게 (레벨 | 이름 | 메시지)
    console_format = '%(levelname)s | %(name)s | %(message)s'

    # 파일: 상세하게 (시간 | 레벨 | 이름 | 함수:라인 | 메시지)
    file_format = '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'

    # ========== 콘솔 핸들러 ==========
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(console_format))
        logger.addHandler(console_handler)

    # ========== 파일 핸들러 ==========
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

        # 날짜별 로그 파일 (예: depth_estimation_20240131.log)
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    # 상위 로거로 전파 방지 (루트 로거에 중복 출력 안 함)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기존 로거 가져오기

    setup_logger로 생성된 로거를 다른 모듈에서 가져올 때 사용.
    setup_logger가 호출된 적 없으면 기본 로거 반환.

    Args:
        name: 로거 이름 (setup_logger에서 사용한 이름)

    Returns:
        logging.Logger: 로거 인스턴스

    사용 예시:
        # main.py에서
        setup_logger('depth_estimation', log_dir=Path('logs'))

        # 다른 모듈에서
        logger = get_logger('depth_estimation.camera')
        logger.info("카메라 초기화")
    """
    return logging.getLogger(name)
