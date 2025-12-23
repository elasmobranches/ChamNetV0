"""
MTL Loss Functions for MMSegmentation

Depth 추정을 위한 손실 함수들을 구현합니다.
- SILogLoss: Scale-Invariant Logarithmic Loss
- L1DepthLoss: L1 Loss with valid mask handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss for Depth Estimation
    
    깊이 추정에서 널리 사용되는 손실 함수입니다.
    스케일에 불변하며, 로그 공간에서 계산됩니다.
    
    수식:
        L = (1/n) * Σ(log(pred) - log(target))² - λ * ((1/n) * Σ(log(pred) - log(target)))²
    
    Args:
        loss_weight (float): 손실 가중치
        lambda_variance (float): 분산 항의 가중치 (기본값: 0.85)
        eps (float): 0으로 나누기 방지 값 (기본값: 1e-6)
        loss_name (str): 손실 이름
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 lambda_variance=0.85,
                 eps=1e-6,
                 loss_name='loss_silog',
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.lambda_variance = lambda_variance
        self.eps = eps
        self._loss_name = loss_name
    
    def forward(self,
                pred,
                target,
                weight=None,
                ignore_index=None,
                **kwargs):
        """Forward 함수
        
        Args:
            pred (Tensor): 예측 깊이 [B, H, W] or [B, 1, H, W]
            target (Tensor): GT 깊이 [B, H, W] or [B, 1, H, W]
            weight (Tensor, optional): 픽셀별 가중치
            ignore_index (int, optional): 무시할 인덱스
        
        Returns:
            Tensor: 손실 값
        """
        # 차원 정규화
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # 유효한 깊이 값 마스크
        valid_mask = (target > self.eps) & torch.isfinite(target)
        
        # ignore_index 처리
        if ignore_index is not None:
            valid_mask = valid_mask & (target != ignore_index)
        
        # weight 처리
        if weight is not None:
            if weight.dim() == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            valid_mask = valid_mask & (weight > 0)
        
        # 유효한 픽셀이 없으면 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 유효한 픽셀만 선택
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # 로그 차이 계산
        log_diff = torch.log(pred_valid.clamp(min=self.eps)) - torch.log(target_valid.clamp(min=self.eps))
        
        # SILog 손실 계산
        mse = (log_diff ** 2).mean()
        mean_log_diff = log_diff.mean()
        
        loss = mse - self.lambda_variance * (mean_log_diff ** 2)
        
        return self.loss_weight * loss
    
    @property
    def loss_name(self):
        """손실 이름 반환"""
        return self._loss_name


@MODELS.register_module()
class L1DepthLoss(nn.Module):
    """L1 Loss for Depth Estimation with Valid Mask
    
    유효한 깊이 값에 대해서만 L1 손실을 계산합니다.
    
    Args:
        loss_weight (float): 손실 가중치
        eps (float): 유효 깊이 판단 임계값 (기본값: 1e-6)
        loss_name (str): 손실 이름
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_l1_depth',
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name
    
    def forward(self,
                pred,
                target,
                weight=None,
                ignore_index=None,
                **kwargs):
        """Forward 함수
        
        Args:
            pred (Tensor): 예측 깊이 [B, H, W] or [B, 1, H, W]
            target (Tensor): GT 깊이 [B, H, W] or [B, 1, H, W]
            weight (Tensor, optional): 픽셀별 가중치
            ignore_index (int, optional): 무시할 인덱스
        
        Returns:
            Tensor: 손실 값
        """
        # 차원 정규화
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # 유효한 깊이 값 마스크
        valid_mask = (target > self.eps) & torch.isfinite(target)
        
        # ignore_index 처리
        if ignore_index is not None:
            valid_mask = valid_mask & (target != ignore_index)
        
        # weight 처리
        if weight is not None:
            if weight.dim() == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            valid_mask = valid_mask & (weight > 0)
        
        # 유효한 픽셀이 없으면 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 유효한 픽셀만 선택
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # L1 손실 계산
        loss = F.l1_loss(pred_valid, target_valid)
        
        return self.loss_weight * loss
    
    @property
    def loss_name(self):
        """손실 이름 반환"""
        return self._loss_name


@MODELS.register_module()
class DepthSmoothLoss(nn.Module):
    """Edge-aware Smoothness Loss for Depth
    
    이미지의 edge 정보를 활용하여 depth map의 smoothness를 유도합니다.
    edge가 없는 곳은 smooth하게, edge가 있는 곳은 sharp하게 유지합니다.
    
    Args:
        loss_weight (float): 손실 가중치
        loss_name (str): 손실 이름
    """
    
    def __init__(self,
                 loss_weight=0.1,
                 loss_name='loss_smooth',
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    
    def forward(self,
                pred,
                image=None,
                **kwargs):
        """Forward 함수
        
        Args:
            pred (Tensor): 예측 깊이 [B, 1, H, W]
            image (Tensor): 원본 이미지 [B, 3, H, W]
        
        Returns:
            Tensor: 손실 값
        """
        if image is None:
            # 이미지가 없으면 일반 smoothness loss
            return self._compute_smoothness(pred)
        
        # Edge-aware smoothness
        return self._compute_edge_aware_smoothness(pred, image)
    
    def _compute_smoothness(self, depth):
        """일반 smoothness loss 계산"""
        # Gradient 계산
        dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        
        return self.loss_weight * (dy.mean() + dx.mean())
    
    def _compute_edge_aware_smoothness(self, depth, image):
        """Edge-aware smoothness loss 계산"""
        # Image gradient
        img_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)
        img_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
        
        # Depth gradient
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        
        # Edge-aware weighting
        weight_y = torch.exp(-img_dy)
        weight_x = torch.exp(-img_dx)
        
        # Weighted smoothness
        smooth_y = (depth_dy * weight_y).mean()
        smooth_x = (depth_dx * weight_x).mean()
        
        return self.loss_weight * (smooth_y + smooth_x)
    
    @property
    def loss_name(self):
        """손실 이름 반환"""
        return self._loss_name


@MODELS.register_module()
class BerHuLoss(nn.Module):
    """BerHu (Reverse Huber) Loss for Depth Estimation
    
    작은 오차에는 L1, 큰 오차에는 L2를 적용하는 손실 함수입니다.
    
    Args:
        loss_weight (float): 손실 가중치
        threshold (float): L1/L2 전환 임계값 (기본값: 0.2)
        loss_name (str): 손실 이름
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 threshold=0.2,
                 loss_name='loss_berhu',
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.threshold = threshold
        self._loss_name = loss_name
    
    def forward(self,
                pred,
                target,
                weight=None,
                ignore_index=None,
                **kwargs):
        """Forward 함수
        
        Args:
            pred (Tensor): 예측 깊이
            target (Tensor): GT 깊이
            weight (Tensor, optional): 픽셀별 가중치
            ignore_index (int, optional): 무시할 인덱스
        
        Returns:
            Tensor: 손실 값
        """
        # 차원 정규화
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        # 유효한 마스크
        valid_mask = (target > 0) & torch.isfinite(target)
        
        if ignore_index is not None:
            valid_mask = valid_mask & (target != ignore_index)
        
        if weight is not None:
            if weight.dim() == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            valid_mask = valid_mask & (weight > 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 유효한 픽셀만
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # 절대 오차
        abs_diff = torch.abs(pred_valid - target_valid)
        
        # 임계값 계산 (데이터 기반)
        c = self.threshold * torch.max(abs_diff).item()
        
        # BerHu loss
        berhu = torch.where(
            abs_diff <= c,
            abs_diff,  # L1
            (abs_diff ** 2 + c ** 2) / (2 * c)  # L2
        )
        
        return self.loss_weight * berhu.mean()
    
    @property
    def loss_name(self):
        """손실 이름 반환"""
        return self._loss_name
