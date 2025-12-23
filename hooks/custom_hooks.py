# custom_hooks.py
# Val Loss 로깅을 위한 Custom Hook (MMSegmentation 1.x / MMEngine 호환)

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class ValLossHook(Hook):
    """
    Validation 중 Loss를 계산하고 TensorBoard에 로깅하는 Hook
    
    MMSegmentation의 기본 Validation은 predict 모드로만 동작하여 Loss를 계산하지 않습니다.
    이 Hook은 Validation의 각 iteration 후에 model을 loss 모드로 다시 forward하여
    Loss를 계산하고 TensorBoard에 기록합니다.
    
    주의:
    - Validation 시 forward pass가 2번 실행됩니다 (predict + loss)
    - Validation 시간이 약 2배 소요될 수 있습니다
    - 하지만 Train Loss와 Val Loss를 함께 모니터링할 수 있습니다
    
    사용법:
        Config 파일에 다음과 같이 추가:
        ```python
        custom_imports = dict(imports=['custom_hooks'], allow_failed_imports=False)
        
        custom_hooks = [
            dict(type='ValLossHook')
        ]
        ```
    """
    
    def __init__(self, log_loss_components: bool = True, **kwargs):
        """
        Args:
            log_loss_components: 개별 loss component도 로깅할지 여부
        """
        super().__init__(**kwargs)
        self.log_loss_components = log_loss_components
        self.val_loss_buffer = []
        self._forward_failed = False  # forward 실패 시 반복 경고 방지
    
    def before_val(self, runner):
        """Validation 시작 전 버퍼 초기화"""
        self.val_loss_buffer.clear()
        self._forward_failed = False
    
    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        """
        Validation의 매 Iteration이 끝날 때마다 호출됩니다.
        model을 loss 모드로 다시 forward하여 loss를 계산합니다.
        """
        # 이미 forward 실패한 경우 스킵
        if self._forward_failed:
            return
        
        try:
            # 모델 가져오기 (DDP wrapper 처리)
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            
            # eval 모드 유지
            model.eval()

            # gradient 계산 없이 loss만 계산 (메모리 효율적)
            with torch.no_grad():
                # forward(mode='loss')를 사용하여 loss 계산
                # 이 방법이 train_step보다 validation에 더 적합함
                # MMSegmentation 모델은 mode='loss'로 forward하면 loss_dict 반환
                loss_dict = model(**data_batch, mode='loss')
            
            # loss_dict에서 'loss'와 개별 component 추출
            losses = {}
            for key, value in loss_dict.items():
                if isinstance(value, dict):
                    # nested dict (e.g., 'log_vars')
                    for sub_key, sub_value in value.items():
                        losses[sub_key] = sub_value
                else:
                    losses[key] = value

            # loss 합계 계산 (for loop 밖으로 이동)
            total_loss = 0.0
            loss_components = {}

            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    loss_val = float(value.item())
                    total_loss += loss_val
                    loss_components[key] = loss_val
                elif isinstance(value, (int, float)):
                    total_loss += float(value)
                    loss_components[key] = float(value)

            # 버퍼에 저장
            self.val_loss_buffer.append({
                'total_loss': total_loss,
                'components': loss_components
            })

            # 첫 번째 배치에서만 로그 출력
            if batch_idx == 0:
                runner.logger.info(
                    f'ValLossHook: First batch loss = {total_loss:.4f}'
                )
                    
        except Exception as e:
            if batch_idx == 0:
                runner.logger.warning(
                    f'ValLossHook: Failed to compute loss: {e}. '
                    f'Val loss logging will be disabled for this epoch.'
                )
            self._forward_failed = True
    
    def after_val_epoch(self, runner, metrics=None):
        """
        Validation epoch이 끝날 때 평균 Loss를 계산하고 로깅합니다.
        
        Args:
            runner: MMEngine Runner 객체
            metrics: Evaluator에서 계산된 메트릭
        """
        if not self.val_loss_buffer:
            if not self._forward_failed:
                runner.logger.warning('ValLossHook: No loss data collected during validation')
            return
        
        # 평균 total loss 계산
        avg_total_loss = sum(item['total_loss'] for item in self.val_loss_buffer) / len(self.val_loss_buffer)
        
        # TensorBoard에 로깅
        runner.message_hub.update_scalar('val/loss', avg_total_loss)
        
        # 개별 loss component 평균 계산 및 로깅
        if self.log_loss_components:
            all_component_names = set()
            for item in self.val_loss_buffer:
                all_component_names.update(item['components'].keys())
            
            avg_components = {}
            for name in all_component_names:
                values = [item['components'].get(name, 0.0) for item in self.val_loss_buffer]
                avg_components[name] = sum(values) / len(values)
                
                # TensorBoard에 개별 component 로깅
                # 'decode.loss_ce' -> 'val/loss_ce' 형태로 변환
                clean_name = name.replace('decode.', '').replace('loss.', '')
                runner.message_hub.update_scalar(f'val/{clean_name}', avg_components[name])
            
            # 터미널 출력
            component_strs = [f"{k}: {v:.4f}" for k, v in avg_components.items()]
            runner.logger.info(
                f'Val Loss: {avg_total_loss:.4f} ({", ".join(component_strs)})'
            )
        else:
            runner.logger.info(f'Val Loss: {avg_total_loss:.4f}')
        
        # 버퍼 초기화
        self.val_loss_buffer.clear()


@HOOKS.register_module()
class DepthValLossHook(ValLossHook):
    """
    Depth Estimation 전용 Val Loss Hook
    
    ValLossHook을 상속받아 depth 특화 로깅을 추가합니다.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def after_val_epoch(self, runner, metrics=None):
        """Depth 전용 로깅 추가"""
        if not self.val_loss_buffer:
            return
        
        # 부모 클래스의 로깅 수행
        super().after_val_epoch(runner, metrics)
        
        # Depth 특화 메트릭 로깅 (metrics에서 가져오기)
        # 주의: metrics는 보통 runner.message_hub에서 가져와야 함
        if metrics is not None and isinstance(metrics, dict):
            depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']
            for metric_name in depth_metrics:
                # prefix 처리 (depth/abs_rel 또는 abs_rel)
                for key in [f'depth/{metric_name}', metric_name]:
                    if key in metrics:
                        runner.logger.info(f'  {metric_name}: {metrics[key]:.4f}')
                        break
