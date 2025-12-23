# hooks/depth_iter_logger_hook.py
# Depth Estimation 전용 CSV Logger Hook

import os
import csv
import time
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class DepthIterLoggerHook(Hook):
    """
    Depth Estimation 전용 CSV Logger

    Train/Val Loss와 Depth 메트릭(abs_rel, rmse, d1 등)을 CSV 파일로 기록합니다.

    Args:
        out_csv (str, optional): CSV 파일 저장 경로. None이면 work_dir/learning_curve.csv
        flush_secs (int): CSV flush 간격 (초). Default: 10
        metrics (list): 기록할 depth 메트릭 리스트.
                       Default: ['abs_rel', 'rmse', 'a1']

    사용법:
        ```python
        # schedule_30kd.py
        custom_hooks = [
            dict(
                type='DepthIterLoggerHook',
                out_csv=None,  # 자동으로 work_dir/learning_curve.csv
                flush_secs=10,
                metrics=['abs_rel', 'rmse', 'a1']
            )
        ]
        ```

    CSV 형식:
        timestamp,global_iter,mode,train_loss,val_loss_avg,val_abs_rel,val_rmse,val_a1
        2025-12-11 08:00:00,0,train,1.234567,,,
        2025-12-11 08:05:30,200,val,,0.123456,0.045678,0.234567,0.876543
    """

    def __init__(self, out_csv=None, flush_secs=10, metrics=None):
        """
        Args:
            out_csv: CSV 파일 경로 (None이면 자동 생성)
            flush_secs: Flush 간격 (초)
            metrics: 기록할 메트릭 리스트 (기본값: ['abs_rel', 'rmse', 'a1'])
        """
        self.out_csv = out_csv
        self.flush_secs = flush_secs
        self._last_flush = time.time()
        self._csv_initialized = False

        # 기본 메트릭 설정
        if metrics is None:
            self.metrics = ['abs_rel', 'rmse', 'a1']
        else:
            self.metrics = metrics

    def before_run(self, runner):
        """학습 시작 전 호출 - CSV 경로 자동 설정 및 초기화"""
        # out_csv가 None이면 work_dir 기반으로 자동 생성
        if self.out_csv is None:
            self.out_csv = os.path.join(runner.work_dir, 'learning_curve.csv')

        # CSV 파일 디렉토리 생성
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)

        # CSV 헤더 생성 (파일이 없으면)
        # 간소화된 포맷: 필수 정보만 (iter, train_loss, depth metrics)
        if not os.path.exists(self.out_csv):
            with open(self.out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                # Depth 전용 헤더 (간소화)
                header = ['iter', 'train_loss']
                # 메트릭 컬럼 추가
                header.extend([f'val_{metric}' for metric in self.metrics])
                writer.writerow(header)

        self._csv_initialized = True
        runner.logger.info(
            f'DepthIterLoggerHook: Saving learning curve to {self.out_csv}\n'
            f'  Metrics: {", ".join(self.metrics)}'
        )

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Train Iteration 종료 후 호출"""
        global_iter = runner.iter

        # Train loss 가져오기
        train_loss = None

        # 방법 1: outputs dict에서 직접 가져오기 (가장 확실)
        if isinstance(outputs, dict) and 'loss' in outputs:
            train_loss = float(outputs['loss'])

        # 방법 2: message_hub에서 현재 loss 가져오기
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

        # 방법 3: log_buffer 확인 (Fallback)
        if train_loss is None and hasattr(runner, 'log_buffer'):
            out = runner.log_buffer.output
            if 'loss' in out:
                try:
                    train_loss = float(out['loss'])
                except Exception:
                    pass

        # loss가 있을 때만 CSV에 기록
        if train_loss is not None:
            self._append_csv(global_iter, train_loss=train_loss)

        # Flush 체크
        now = time.time()
        if now - self._last_flush > self.flush_secs:
            self._last_flush = now

    def after_val_epoch(self, runner, metrics=None):
        """Validation Epoch 종료 후 호출 - 평균 Loss 및 Depth Metrics 기록"""
        global_iter = runner.iter

        # 1. Val Loss를 message_hub에서 가져오기 (DepthValLossHook이 저장한 값)
        val_loss_avg = None
        if hasattr(runner, 'message_hub'):
            try:
                # DepthValLossHook이 'val/loss'에 저장함
                loss_buffer = runner.message_hub.get_scalar('val/loss')
                if loss_buffer is not None:
                    if hasattr(loss_buffer, 'current'):
                        val_loss_avg = float(loss_buffer.current())
                    elif hasattr(loss_buffer, 'mean'):
                        val_loss_avg = float(loss_buffer.mean())
            except Exception as e:
                runner.logger.debug(f'Failed to get val/loss from message_hub: {e}')

        # 2. Depth 메트릭들 가져오기
        metric_values = {}

        if hasattr(runner, 'message_hub'):
            for metric_name in self.metrics:
                try:
                    # 'depth/abs_rel' 또는 'abs_rel' 형태로 시도
                    for key in [f'depth/{metric_name}', metric_name]:
                        metric_buffer = runner.message_hub.get_scalar(key)
                        if metric_buffer is not None:
                            if hasattr(metric_buffer, 'current'):
                                metric_values[metric_name] = float(metric_buffer.current())
                            elif hasattr(metric_buffer, 'mean'):
                                metric_values[metric_name] = float(metric_buffer.mean())
                            break
                except Exception:
                    continue

        # Fallback: metrics 파라미터에서 가져오기
        if not metric_values and metrics is not None and isinstance(metrics, dict):
            for metric_name in self.metrics:
                for key in [f'depth/{metric_name}', metric_name]:
                    if key in metrics:
                        metric_values[metric_name] = float(metrics[key])
                        break

        # CSV에 기록 (val_loss는 ValLossHook 비활성화로 제거)
        self._append_csv(global_iter, train_loss=None, metric_values=metric_values)

        # 터미널 로깅
        val_loss_str = f'{val_loss_avg:.4f}' if val_loss_avg is not None else 'N/A'
        metric_strs = []
        for metric_name in self.metrics:
            if metric_name in metric_values:
                metric_strs.append(f'{metric_name}={metric_values[metric_name]:.4f}')
            else:
                metric_strs.append(f'{metric_name}=N/A')

        runner.logger.info(
            f'DepthIterLoggerHook: Validation @ iter {global_iter} - '
            f'Val Loss: {val_loss_str}, {", ".join(metric_strs)}'
        )

    def _append_csv(self, global_iter, train_loss=None, metric_values=None):
        """간소화된 CSV 기록 - 필수 정보만 (iter, train_loss, depth metrics)"""
        # None이면 빈 문자열로 처리
        str_train = f"{train_loss:.6f}" if train_loss is not None else ""

        # 메트릭 값들 처리
        metric_strs = []
        if metric_values is None:
            metric_values = {}

        for metric_name in self.metrics:
            if metric_name in metric_values:
                metric_strs.append(f"{metric_values[metric_name]:.6f}")
            else:
                metric_strs.append("")

        # Row 구성 (간소화)
        row = [global_iter, str_train]
        row.extend(metric_strs)

        # CSV에 기록
        with open(self.out_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
