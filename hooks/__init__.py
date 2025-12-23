# hooks/__init__.py
# Custom hooks를 명시적으로 import하여 registry에 등록

from .iter_logger_hook import IterLoggerHook
from .mtl_iter_logger_hook import MTLIterLoggerHook
from .depth_iter_logger_hook import DepthIterLoggerHook
from .custom_hooks import ValLossHook, DepthValLossHook

__all__ = [
    'IterLoggerHook',           # Segmentation용
    'MTLIterLoggerHook',        # MTL용
    'DepthIterLoggerHook',      # Depth 전용
    'ValLossHook',
    'DepthValLossHook'
]

