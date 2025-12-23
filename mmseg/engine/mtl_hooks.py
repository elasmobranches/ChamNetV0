"""
MTL Custom Hooks for MMSegmentation

Multi-Task Learning을 위한 커스텀 훅들을 구현합니다.
- MTL 메트릭 로깅
- 가중치 변화 추적
- 시각화
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.registry import HOOKS


@HOOKS.register_module()
class MTLMetricsHook(Hook):
    """MTL 메트릭 로깅 훅
    
    Segmentation과 Depth 태스크의 메트릭을 추적하고 로깅합니다.
    
    Args:
        log_interval (int): 로깅 간격 (iterations)
        metrics (List[str]): 추적할 메트릭 리스트
    """
    
    def __init__(self,
                 log_interval: int = 50,
                 metrics: List[str] = ['seg_miou', 'depth_absrel', 'depth_rmse', 'depth_delta1']):
        self.log_interval = log_interval
        self.metrics = metrics
        self.metric_history = {metric: [] for metric in metrics}
    
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """학습 iteration 후 메트릭 로깅"""
        
        if runner.iter % self.log_interval == 0:
            # 손실 값 로깅
            if outputs is not None:
                # Segmentation 손실
                seg_losses = {k: v for k, v in outputs.items() if 'seg' in k}
                for name, value in seg_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    runner.logger.info(f"{name}: {value:.4f}")
                
                # Depth 손실
                depth_losses = {k: v for k, v in outputs.items() if 'depth' in k}
                for name, value in depth_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    runner.logger.info(f"{name}: {value:.4f}")
                
                # MTL 가중치 (있는 경우)
                if 'dwa_weight_seg' in outputs:
                    runner.logger.info(f"DWA weights - Seg: {outputs['dwa_weight_seg']:.4f}, "
                                     f"Depth: {outputs['dwa_weight_depth']:.4f}")
                
                if 'log_var_seg' in outputs:
                    runner.logger.info(f"Uncertainty weights - "
                                     f"log_var_seg: {outputs['log_var_seg']:.4f}, "
                                     f"log_var_depth: {outputs['log_var_depth']:.4f}")
    
    def after_val_iter(self,
                      runner: Runner,
                      batch_idx: int,
                      data_batch: dict = None,
                      outputs: Sequence = None) -> None:
        """검증 iteration 후 메트릭 계산"""
        
        if outputs is None:
            return
        
        # 예측과 GT 추출
        for output in outputs:
            # Segmentation 메트릭
            if 'pred_sem_seg' in output and 'gt_sem_seg' in output:
                pred_seg = output['pred_sem_seg']['data']
                gt_seg = output['gt_sem_seg']['data']
                
                # mIoU 계산 (간단한 버전)
                if 'seg_miou' in self.metrics:
                    miou = self._compute_miou(pred_seg, gt_seg)
                    self.metric_history['seg_miou'].append(miou)
            
            # Depth 메트릭
            if 'pred_depth' in output and 'gt_depth' in output:
                pred_depth = output['pred_depth']['data']
                gt_depth = output['gt_depth']['data']
                
                # Depth 메트릭 계산
                depth_metrics = self._compute_depth_metrics(pred_depth, gt_depth)
                for metric_name, value in depth_metrics.items():
                    if metric_name in self.metrics:
                        self.metric_history[metric_name].append(value)
    
    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict] = None) -> None:
        """검증 epoch 후 평균 메트릭 로깅"""
        
        for metric_name, values in self.metric_history.items():
            if len(values) > 0:
                avg_value = np.mean(values)
                runner.logger.info(f"Val {metric_name}: {avg_value:.4f}")
                
                # 메트릭 리셋
                self.metric_history[metric_name] = []
    
    def _compute_miou(self, pred: torch.Tensor, target: torch.Tensor, 
                     num_classes: int = 7) -> float:
        """간단한 mIoU 계산"""
        ious = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                ious.append((intersection / union).item())
        
        return np.mean(ious) if ious else 0.0
    
    def _compute_depth_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Depth 메트릭 계산"""
        # 유효한 마스크
        valid_mask = (target > 0) & torch.isfinite(target)
        
        if valid_mask.sum() == 0:
            return {}
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # AbsRel: |pred - target| / target
        abs_rel = ((pred_valid - target_valid).abs() / target_valid).mean().item()
        
        # RMSE
        rmse = torch.sqrt(((pred_valid - target_valid) ** 2).mean()).item()
        
        # Delta < 1.25
        threshold = 1.25
        ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
        delta1 = (ratio < threshold).float().mean().item()
        
        return {
            'depth_absrel': abs_rel,
            'depth_rmse': rmse,
            'depth_delta1': delta1
        }


@HOOKS.register_module()
class MTLWeightLoggerHook(Hook):
    """MTL 가중치 로깅 훅
    
    DWA 또는 Uncertainty Weighting의 가중치 변화를 추적합니다.
    
    Args:
        log_interval (int): 로깅 간격
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.weight_history = {'seg': [], 'depth': []}
    
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """학습 iteration 후 가중치 로깅"""
        
        if runner.iter % self.log_interval != 0:
            return
        
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # DWA 가중치
        if hasattr(model, 'dwa_weights'):
            seg_weight, depth_weight = model.dwa_weights
            self.weight_history['seg'].append(seg_weight)
            self.weight_history['depth'].append(depth_weight)
            
            runner.logger.info(f"[Iter {runner.iter}] DWA Weights - "
                             f"Seg: {seg_weight:.4f}, Depth: {depth_weight:.4f}")
        
        # Uncertainty 가중치
        elif hasattr(model, 'log_var_seg') and hasattr(model, 'log_var_depth'):
            log_var_seg = model.log_var_seg.item()
            log_var_depth = model.log_var_depth.item()
            
            # 실제 가중치 계산
            seg_weight = 1.0 / (2 * np.exp(log_var_seg))
            depth_weight = 1.0 / (2 * np.exp(log_var_depth))
            
            self.weight_history['seg'].append(seg_weight)
            self.weight_history['depth'].append(depth_weight)
            
            runner.logger.info(f"[Iter {runner.iter}] Uncertainty Weights - "
                             f"Seg: {seg_weight:.4f} (log_var: {log_var_seg:.4f}), "
                             f"Depth: {depth_weight:.4f} (log_var: {log_var_depth:.4f})")
    
    def after_run(self, runner: Runner) -> None:
        """학습 종료 후 가중치 히스토리 저장"""
        
        if len(self.weight_history['seg']) > 0:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            iterations = list(range(0, len(self.weight_history['seg']) * self.log_interval, 
                                  self.log_interval))
            
            ax.plot(iterations, self.weight_history['seg'], label='Segmentation', linewidth=2)
            ax.plot(iterations, self.weight_history['depth'], label='Depth', linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Task Weight')
            ax.set_title('MTL Task Weights over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 저장
            save_path = os.path.join(runner.work_dir, 'mtl_weights.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            runner.logger.info(f"MTL weight history saved to {save_path}")


@HOOKS.register_module()
class MTLVisualizationHook(Hook):
    """MTL 시각화 훅
    
    Segmentation과 Depth 예측을 시각화합니다.
    
    Args:
        draw (bool): 시각화 활성화 여부
        interval (int): 시각화 간격
        max_samples (int): 최대 샘플 수
        show_seg (bool): Segmentation 시각화 여부
        show_depth (bool): Depth 시각화 여부
        show_gt (bool): GT 시각화 여부
    """
    
    def __init__(self,
                 draw: bool = True,
                 interval: int = 500,
                 max_samples: int = 4,
                 show_seg: bool = True,
                 show_depth: bool = True,
                 show_gt: bool = True):
        self.draw = draw
        self.interval = interval
        self.max_samples = max_samples
        self.show_seg = show_seg
        self.show_depth = show_depth
        self.show_gt = show_gt
    
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """학습 중 시각화"""
        
        if not self.draw or runner.iter % self.interval != 0:
            return
        
        self._visualize(runner, data_batch, outputs, phase='train')
    
    def after_val_iter(self,
                      runner: Runner,
                      batch_idx: int,
                      data_batch: dict = None,
                      outputs: Sequence = None) -> None:
        """검증 중 시각화"""
        
        if not self.draw or batch_idx >= self.max_samples:
            return
        
        self._visualize(runner, data_batch, outputs, phase='val')
    
    def _visualize(self, runner: Runner, data_batch: dict, 
                  outputs: Union[dict, Sequence], phase: str):
        """실제 시각화 수행"""
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # 색상 팔레트 (7개 클래스)
        palette = np.array([
            [0, 0, 0],      # background (검정)
            [128, 0, 0],    # class 1 (짙은 빨강)
            [0, 128, 0],    # class 2 (짙은 초록)
            [128, 128, 0],  # class 3 (올리브)
            [0, 0, 128],    # class 4 (짙은 파랑)
            [128, 0, 128],  # class 5 (보라)
            [0, 128, 128],  # class 6 (청록)
        ])
        
        # 출력 디렉토리
        vis_dir = os.path.join(runner.work_dir, 'visualizations', phase)
        os.makedirs(vis_dir, exist_ok=True)
        
        # 데이터 추출
        if isinstance(outputs, dict):
            outputs = [outputs]
        
        for idx, output in enumerate(outputs[:self.max_samples]):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 원본 이미지
            if 'img' in data_batch:
                img = data_batch['img'][idx].cpu().numpy()
                if img.shape[0] == 3:  # CHW -> HWC
                    img = img.transpose(1, 2, 0)
                # 정규화 해제
                img = img * np.array([58.395, 57.12, 57.375]) + np.array([123.675, 116.28, 103.53])
                img = np.clip(img, 0, 255).astype(np.uint8)
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Input Image')
                axes[0, 0].axis('off')
            
            # Segmentation 예측
            if self.show_seg and 'pred_sem_seg' in output:
                pred_seg = output['pred_sem_seg']['data'].cpu().numpy()
                if pred_seg.ndim == 3:
                    pred_seg = pred_seg[0]
                
                # 색상 매핑
                seg_colored = palette[pred_seg]
                axes[0, 1].imshow(seg_colored)
                axes[0, 1].set_title('Predicted Segmentation')
                axes[0, 1].axis('off')
            
            # Segmentation GT
            if self.show_gt and self.show_seg and 'gt_sem_seg' in output:
                gt_seg = output['gt_sem_seg']['data'].cpu().numpy()
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                
                gt_seg_colored = palette[gt_seg]
                axes[0, 2].imshow(gt_seg_colored)
                axes[0, 2].set_title('GT Segmentation')
                axes[0, 2].axis('off')
            
            # Depth 예측
            if self.show_depth and 'pred_depth' in output:
                pred_depth = output['pred_depth']['data'].cpu().numpy()
                if pred_depth.ndim == 3:
                    pred_depth = pred_depth[0]
                
                im = axes[1, 0].imshow(pred_depth, cmap='viridis')
                axes[1, 0].set_title('Predicted Depth')
                axes[1, 0].axis('off')
                plt.colorbar(im, ax=axes[1, 0])
            
            # Depth GT
            if self.show_gt and self.show_depth and 'gt_depth' in output:
                gt_depth = output['gt_depth']['data'].cpu().numpy()
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth[0]
                
                im = axes[1, 1].imshow(gt_depth, cmap='viridis')
                axes[1, 1].set_title('GT Depth')
                axes[1, 1].axis('off')
                plt.colorbar(im, ax=axes[1, 1])
            
            # Depth 오차 맵
            if self.show_depth and 'pred_depth' in output and 'gt_depth' in output:
                error_map = np.abs(pred_depth - gt_depth)
                im = axes[1, 2].imshow(error_map, cmap='hot')
                axes[1, 2].set_title('Depth Error Map')
                axes[1, 2].axis('off')
                plt.colorbar(im, ax=axes[1, 2])
            
            plt.tight_layout()
            
            # 저장
            save_path = os.path.join(vis_dir, f'iter_{runner.iter}_sample_{idx}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        runner.logger.info(f"Visualizations saved to {vis_dir}")
