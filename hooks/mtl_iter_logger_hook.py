# hooks/mtl_iter_logger_hook.py
# MTL (Multi-Task Learning) Train/Val Loss와 Metrics를 CSV 파일로 기록하는 Hook

import os
import csv
import time
import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class MTLIterLoggerHook(Hook):
    """
    MTL (Multi-Task Learning) 전용 CSV Logger Hook

    Segmentation과 Depth 태스크의 메트릭을 CSV에 기록합니다:
    - Train iteration마다: train_loss, mtl_weight_seg, mtl_weight_depth, dwa_weight_seg, dwa_weight_depth
    - Val epoch마다: val_mIoU (Seg), val_abs_rel (Depth)

    Args:
        out_csv (str): CSV 파일 저장 경로 (None이면 자동으로 work_dir/learning_curve.csv)
        flush_secs (int): CSV flush 간격 (초)

    CSV 형식:
        iter, train_loss, val_mIoU, val_abs_rel,
        mtl_weight_seg, mtl_weight_depth,
        dwa_weight_seg, dwa_weight_depth

    사용법:
        ```python
        custom_imports = dict(
            imports=['hooks.mtl_iter_logger_hook'],
            allow_failed_imports=False
        )

        custom_hooks = [
            dict(
                type='MTLIterLoggerHook',
                out_csv=None,  # 자동으로 work_dir/learning_curve.csv
                flush_secs=10
            )
        ]
        ```
    """

    def __init__(self, out_csv=None, flush_secs=10):
        self.out_csv = out_csv
        self.flush_secs = flush_secs
        self._last_flush = time.time()
        self._csv_initialized = False
        self._row_buffer = {}  # 현재 iteration의 데이터를 임시 저장

    def before_run(self, runner):
        """학습 시작 전 호출 - CSV 경로 자동 설정 및 초기화"""
        # out_csv가 None이면 work_dir 기반으로 자동 생성
        if self.out_csv is None:
            self.out_csv = os.path.join(runner.work_dir, 'learning_curve.csv')

        # CSV 파일 디렉토리 생성
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)

        # CSV 헤더 생성 (파일이 없으면)
        # MTL 전용: 8개 컬럼
        if not os.path.exists(self.out_csv):
            with open(self.out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iter', 'train_loss',
                    'val_mIoU', 'val_abs_rel',
                    'mtl_weight_seg', 'mtl_weight_depth',
                    'dwa_weight_seg', 'dwa_weight_depth'
                ])

        self._csv_initialized = True
        runner.logger.info(f'MTLIterLoggerHook: Saving MTL learning curve to {self.out_csv}')

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Train Iteration 종료 후 호출 - Loss와 MTL 가중치 기록"""
        global_iter = runner.iter

        # 새로운 iteration이면 buffer 초기화 및 이전 row 저장
        if global_iter not in self._row_buffer:
            # 이전 iteration 데이터가 있으면 CSV에 기록
            if self._row_buffer:
                self._flush_buffer()

            # 새 buffer 초기화
            self._row_buffer = {
                'iter': global_iter,
                'train_loss': None,
                'val_mIoU': None,
                'val_abs_rel': None,
                'mtl_weight_seg': None,
                'mtl_weight_depth': None,
                'dwa_weight_seg': None,
                'dwa_weight_depth': None,
            }

        # 데이터 추출
        train_loss = None
        mtl_weight_seg = None
        mtl_weight_depth = None
        dwa_weight_seg = None
        dwa_weight_depth = None

        # outputs dict에서 직접 가져오기
        if isinstance(outputs, dict):
            # Total loss
            if 'loss' in outputs:
                train_loss = float(outputs['loss'])

            # MTL 가중치 (Uncertainty Weighting)
            if 'mtl_weight_seg' in outputs:
                mtl_weight_seg = float(outputs['mtl_weight_seg'])
            if 'mtl_weight_depth' in outputs:
                mtl_weight_depth = float(outputs['mtl_weight_depth'])

            # DWA 가중치
            if 'dwa_weight_seg' in outputs:
                dwa_weight_seg = float(outputs['dwa_weight_seg'])
            if 'dwa_weight_depth' in outputs:
                dwa_weight_depth = float(outputs['dwa_weight_depth'])

        # message_hub에서 가져오기 (Fallback)
        elif hasattr(runner, 'message_hub'):
            try:
                loss_buffer = runner.message_hub.get_scalar('train/loss')
                if loss_buffer is not None:
                    if hasattr(loss_buffer, 'current'):
                        train_loss = float(loss_buffer.current())
                    elif hasattr(loss_buffer, 'mean'):
                        train_loss = float(loss_buffer.mean())
            except Exception:
                pass

        # log_buffer 확인 (Fallback)
        if train_loss is None and hasattr(runner, 'log_buffer'):
            out = runner.log_buffer.output
            if 'loss' in out:
                try:
                    train_loss = float(out['loss'])
                except Exception:
                    pass

        # Buffer에 train 데이터 저장
        if train_loss is not None:
            self._row_buffer['train_loss'] = train_loss
        if mtl_weight_seg is not None:
            self._row_buffer['mtl_weight_seg'] = mtl_weight_seg
        if mtl_weight_depth is not None:
            self._row_buffer['mtl_weight_depth'] = mtl_weight_depth
        if dwa_weight_seg is not None:
            self._row_buffer['dwa_weight_seg'] = dwa_weight_seg
        if dwa_weight_depth is not None:
            self._row_buffer['dwa_weight_depth'] = dwa_weight_depth

        # Flush 체크
        now = time.time()
        if now - self._last_flush > self.flush_secs:
            self._last_flush = now

    def after_val_epoch(self, runner, metrics=None):
        """Validation Epoch 종료 후 호출 - Seg mIoU, Depth abs_rel 기록"""
        global_iter = runner.iter

        # Validation 메트릭 가져오기
        val_miou = None
        val_abs_rel = None

        # Method 1: metrics 파라미터에서 먼저 가져오기 (가장 확실!)
        # MMEngine은 Evaluator 결과를 metrics dict로 전달함
        if metrics is not None and isinstance(metrics, dict):
            # Segmentation mIoU
            for key in ['mIoU', 'seg/mIoU', 'val/mIoU', 'IoUMetric/mIoU']:
                if key in metrics:
                    val_miou = float(metrics[key])
                    break

            # Depth abs_rel
            for key in ['depth/abs_rel', 'abs_rel']:
                if key in metrics:
                    val_abs_rel = float(metrics[key])
                    break

        # Method 2: message_hub에서 가져오기 (Fallback)
        if val_miou is None and val_abs_rel is None and hasattr(runner, 'message_hub'):
            try:
                # Segmentation mIoU
                for key in ['mIoU', 'seg/mIoU', 'val/mIoU']:
                    try:
                        metric_buffer = runner.message_hub.get_scalar(key)
                        if metric_buffer is not None:
                            if hasattr(metric_buffer, 'current'):
                                val_miou = float(metric_buffer.current())
                                break
                    except (KeyError, AttributeError):
                        continue

                # Depth abs_rel
                for key in ['depth/abs_rel', 'abs_rel']:
                    try:
                        metric_buffer = runner.message_hub.get_scalar(key)
                        if metric_buffer is not None:
                            if hasattr(metric_buffer, 'current'):
                                val_abs_rel = float(metric_buffer.current())
                                break
                    except (KeyError, AttributeError):
                        continue
            except Exception as e:
                runner.logger.debug(f'Failed to get metrics from message_hub: {e}')

        # 디버그 로깅
        if val_miou is not None or val_abs_rel is not None:
            msg_parts = []
            if val_miou is not None:
                msg_parts.append(f'mIoU={val_miou:.4f}')
            if val_abs_rel is not None:
                msg_parts.append(f'abs_rel={val_abs_rel:.4f}')
            runner.logger.info(f'MTLIterLoggerHook: Validation @ iter {global_iter} - {", ".join(msg_parts)}')
        else:
            # metrics 내용 출력하여 디버깅
            keys_str = list(metrics.keys()) if metrics else "None"
            runner.logger.warning(f'MTLIterLoggerHook: Could not find mIoU/abs_rel. Available keys: {keys_str}')

        # Buffer에 validation 데이터 추가 (같은 iteration의 데이터를 합침)
        if global_iter in self._row_buffer:
            if val_miou is not None:
                self._row_buffer['val_mIoU'] = val_miou
            if val_abs_rel is not None:
                self._row_buffer['val_abs_rel'] = val_abs_rel
        else:
            # Buffer에 없으면 새로 생성 (validation만 있는 경우)
            self._row_buffer = {
                'iter': global_iter,
                'train_loss': None,
                'val_mIoU': val_miou,
                'val_abs_rel': val_abs_rel,
                'mtl_weight_seg': None,
                'mtl_weight_depth': None,
                'dwa_weight_seg': None,
                'dwa_weight_depth': None,
            }

    def after_run(self, runner):
        """학습 종료 후 호출 - 남은 buffer 데이터 flush"""
        if self._row_buffer:
            self._flush_buffer()

    def _flush_buffer(self):
        """Buffer의 데이터를 CSV에 기록"""
        if not self._row_buffer or 'iter' not in self._row_buffer:
            return

        # None이면 빈 문자열로 처리
        def fmt(val):
            return f"{val:.6f}" if val is not None else ""

        row = [
            self._row_buffer['iter'],
            fmt(self._row_buffer.get('train_loss')),
            fmt(self._row_buffer.get('val_mIoU')),
            fmt(self._row_buffer.get('val_abs_rel')),
            fmt(self._row_buffer.get('mtl_weight_seg')),
            fmt(self._row_buffer.get('mtl_weight_depth')),
            fmt(self._row_buffer.get('dwa_weight_seg')),
            fmt(self._row_buffer.get('dwa_weight_depth')),
        ]

        with open(self.out_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
