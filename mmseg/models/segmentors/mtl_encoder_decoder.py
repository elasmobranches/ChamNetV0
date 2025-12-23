"""
Multi-Task Learning Encoder-Decoder for MMSegmentation

이 모듈은 SegFormer 기반 MTL 모델을 구현합니다.
단일 인코더(backbone)를 공유하고, Segmentation과 Depth 각각의 디코더 헤드를 가집니다.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.structures import PixelData
from mmseg.models import builder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList, SampleList, add_prefix


@MODELS.register_module()
class MTLEncoderDecoder(EncoderDecoder):
    """Multi-Task Learning Encoder-Decoder
    
    단일 인코더(backbone)를 공유하여 Segmentation과 Depth를 동시에 학습합니다.
    DWA(Dynamic Weight Average)와 Uncertainty Weighting을 지원합니다.
    
    Args:
        backbone (ConfigType): Shared backbone 설정
        seg_decode_head (ConfigType): Segmentation decoder head 설정
        depth_decode_head (ConfigType): Depth decoder head 설정
        mtl_config (dict): MTL 설정 (weight strategy, DWA, uncertainty 등)
        auxiliary_head (OptConfigType): Auxiliary head 설정 (optional)
        train_cfg (OptConfigType): Training 설정
        test_cfg (OptConfigType): Testing 설정
        data_preprocessor (OptConfigType): Data preprocessor 설정
        pretrained (str): Pretrained model 경로 (deprecated)
        init_cfg (OptMultiConfig): Weight 초기화 설정
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 seg_decode_head: ConfigType,
                 depth_decode_head: ConfigType,
                 mtl_config: dict,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        
        # 기본 EncoderDecoder 초기화 (segmentation head만으로)
        super().__init__(
            backbone=backbone,
            decode_head=seg_decode_head,  # seg_decode_head를 기본 decode_head로 사용
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        
        # Segmentation head 저장 (기본 decode_head를 seg_decode_head로 참조)
        self.seg_decode_head = self.decode_head
        
        # Depth head 추가
        self.depth_decode_head = builder.build_head(depth_decode_head)
        self.align_corners = self.seg_decode_head.align_corners
        self.num_classes = self.seg_decode_head.num_classes
        
        # MTL 설정
        self.mtl_config = mtl_config
        self.weight_strategy = mtl_config.get('weight_strategy', 'fixed')
        
        # 손실 가중치 초기화
        self._init_loss_weights()
        
        # DWA (Dynamic Weight Average) 초기화
        if self.weight_strategy == 'dwa' and mtl_config['dwa_config']['enabled']:
            self._init_dwa()
        
        # Uncertainty Weighting 초기화
        elif self.weight_strategy == 'uncertainty' and mtl_config['uncertainty_config']['enabled']:
            self._init_uncertainty_weighting()
        
        # Fixed weights
        else:
            self.seg_weight = mtl_config['fixed_weights']['seg_weight']
            self.depth_weight = mtl_config['fixed_weights']['depth_weight']
        
        # 메트릭 추적을 위한 변수
        self.loss_history = {'seg': [], 'depth': []}
        self.current_iter = 0
    
    def _init_loss_weights(self):
        """손실 가중치 초기화"""
        self.seg_loss_weight = self.mtl_config.get('seg_loss_weight', 1.0)
        self.depth_loss_weight = self.mtl_config.get('depth_loss_weight', 1.0)
    
    def _init_dwa(self):
        """DWA (Dynamic Weight Average) 초기화"""
        dwa_config = self.mtl_config['dwa_config']
        self.dwa_enabled = True
        self.dwa_num_tasks = dwa_config['num_tasks']
        self.dwa_temperature = dwa_config['temperature']
        self.dwa_window_size = dwa_config['window_size']
        self.dwa_update_freq = dwa_config['update_freq']
        
        # DWA 가중치 초기화
        self.dwa_weights = [1.0 / self.dwa_num_tasks] * self.dwa_num_tasks
        
        # 손실 히스토리
        self.dwa_loss_history = [[] for _ in range(self.dwa_num_tasks)]
        
        print(f"✅ DWA initialized: temperature={self.dwa_temperature}, "
              f"window_size={self.dwa_window_size}, update_freq={self.dwa_update_freq}")
    
    def _init_uncertainty_weighting(self):
        """Uncertainty Weighting 초기화 (Kendall et al., CVPR 2018)"""
        unc_config = self.mtl_config['uncertainty_config']
        self.uncertainty_enabled = True
        
        # 학습 가능한 log variance 파라미터
        self.log_var_seg = nn.Parameter(
            torch.tensor([unc_config['init_log_var_seg']], dtype=torch.float32)
        )
        self.log_var_depth = nn.Parameter(
            torch.tensor([unc_config['init_log_var_depth']], dtype=torch.float32)
        )
        
        print(f"✅ Uncertainty Weighting initialized: "
              f"log_var_seg={unc_config['init_log_var_seg']}, "
              f"log_var_depth={unc_config['init_log_var_depth']}")
    
    def update_dwa_weights(self, seg_loss: float, depth_loss: float):
        """DWA 가중치 업데이트"""
        if not hasattr(self, 'dwa_enabled') or not self.dwa_enabled:
            return
        
        # 손실 히스토리에 추가
        self.dwa_loss_history[0].append(seg_loss)
        self.dwa_loss_history[1].append(depth_loss)
        
        # 윈도우 크기 유지
        for i in range(self.dwa_num_tasks):
            if len(self.dwa_loss_history[i]) > self.dwa_window_size:
                self.dwa_loss_history[i].pop(0)
        
        # 충분한 히스토리가 쌓이지 않았으면 스킵
        if any(len(history) < 2 for history in self.dwa_loss_history):
            return
        
        # 손실 변화율 계산
        loss_ratios = []
        for i in range(self.dwa_num_tasks):
            # 최근 손실 평균
            recent_avg = sum(self.dwa_loss_history[i][-3:]) / min(3, len(self.dwa_loss_history[i]))
            # 이전 손실 평균
            prev_avg = sum(self.dwa_loss_history[i][:-1]) / max(1, len(self.dwa_loss_history[i]) - 1)
            
            if prev_avg > 1e-8:
                ratio = recent_avg / prev_avg
            else:
                ratio = 1.0
            loss_ratios.append(ratio)
        
        # Softmax를 사용하여 가중치 계산
        import numpy as np
        loss_ratios = np.array(loss_ratios)
        exp_weights = np.exp(loss_ratios / self.dwa_temperature)
        weights = self.dwa_num_tasks * exp_weights / np.sum(exp_weights)
        
        self.dwa_weights = weights.tolist()
    
    def extract_feat(self, inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """백본으로부터 특징 추출
        
        Args:
            inputs (Tensor): 입력 이미지 [B, C, H, W]
        
        Returns:
            Union[Tensor, Tuple[Tensor]]: 추출된 특징
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, inputs: torch.Tensor, 
                     batch_img_metas: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """인코딩 및 디코딩 (inference용)
        
        Args:
            inputs (Tensor): 입력 이미지
            batch_img_metas (List[dict]): 이미지 메타데이터
        
        Returns:
            Tuple[Tensor, Tensor]: (seg_logits, depth_pred)
        """
        x = self.extract_feat(inputs)
        seg_logits = self.seg_decode_head.predict(x, batch_img_metas, self.test_cfg)
        depth_pred = self.depth_decode_head.predict(x, batch_img_metas, self.test_cfg)
        
        return seg_logits, depth_pred
    
    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:
        """손실 계산
        
        Args:
            inputs (Tensor): 입력 이미지
            data_samples (SampleList): 데이터 샘플 (GT 포함)
        
        Returns:
            dict: 손실 딕셔너리
        """
        x = self.extract_feat(inputs)
        
        losses = dict()
        
        # Segmentation 손실
        seg_loss_decode = self.seg_decode_head.loss(x, data_samples, self.train_cfg)
        losses.update(add_prefix(seg_loss_decode, 'seg'))
        
        # Depth 손실
        depth_loss_decode = self.depth_decode_head.loss(x, data_samples, self.train_cfg)
        losses.update(add_prefix(depth_loss_decode, 'depth'))
        
        # Auxiliary head 손실
        if self.with_auxiliary_head:
            loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        
        # MTL 가중치 적용
        self._apply_mtl_weights(losses)
        
        return losses
    
    def _apply_mtl_weights(self, losses: dict):
        """MTL 가중치 적용
        
        Args:
            losses (dict): 손실 딕셔너리 (in-place 수정)
        """
        # 현재 iteration 증가
        self.current_iter += 1
        
        # Uncertainty Weighting
        if hasattr(self, 'uncertainty_enabled') and self.uncertainty_enabled:
            # Segmentation 손실 가중치
            seg_loss = sum(v for k, v in losses.items() if 'seg' in k)
            weighted_seg_loss = seg_loss / (2 * torch.exp(self.log_var_seg)) + self.log_var_seg / 2
            
            # Depth 손실 가중치
            depth_loss = sum(v for k, v in losses.items() if 'depth' in k)
            weighted_depth_loss = depth_loss / (2 * torch.exp(self.log_var_depth)) + self.log_var_depth / 2
            
            # 전체 손실
            losses['loss'] = weighted_seg_loss + weighted_depth_loss
            
            # 로깅용
            losses['log_var_seg'] = self.log_var_seg.detach()
            losses['log_var_depth'] = self.log_var_depth.detach()
        
        # DWA
        elif hasattr(self, 'dwa_enabled') and self.dwa_enabled:
            # DWA 가중치 업데이트 (주기적으로)
            if self.current_iter % self.dwa_update_freq == 0:
                seg_loss = sum(v.item() for k, v in losses.items() if 'seg' in k)
                depth_loss = sum(v.item() for k, v in losses.items() if 'depth' in k)
                self.update_dwa_weights(seg_loss, depth_loss)
            
            # 가중치 적용
            seg_weight, depth_weight = self.dwa_weights
            
            seg_loss = sum(v for k, v in losses.items() if 'seg' in k)
            depth_loss = sum(v for k, v in losses.items() if 'depth' in k)
            
            losses['loss'] = seg_weight * seg_loss + depth_weight * depth_loss
            
            # 로깅용
            losses['dwa_weight_seg'] = torch.tensor(seg_weight)
            losses['dwa_weight_depth'] = torch.tensor(depth_weight)
        
        # Fixed weights
        else:
            seg_loss = sum(v for k, v in losses.items() if 'seg' in k)
            depth_loss = sum(v for k, v in losses.items() if 'depth' in k)
            
            losses['loss'] = self.seg_weight * seg_loss + self.depth_weight * depth_loss
    
    def predict(self, inputs: torch.Tensor, data_samples: OptSampleList = None) -> SampleList:
        """예측 (inference)
        
        Args:
            inputs (Tensor): 입력 이미지
            data_samples (OptSampleList): 데이터 샘플
        
        Returns:
            SampleList: 예측 결과
        """
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict()] * inputs.shape[0]
        
        seg_logits, depth_pred = self.encode_decode(inputs, batch_img_metas)
        
        return self.postprocess_result(seg_logits, depth_pred, data_samples)
    
    def postprocess_result(self, seg_logits: torch.Tensor, depth_pred: torch.Tensor,
                          data_samples: OptSampleList) -> SampleList:
        """결과 후처리
        
        Args:
            seg_logits (Tensor): Segmentation logits
            depth_pred (Tensor): Depth predictions
            data_samples (OptSampleList): 데이터 샘플
        
        Returns:
            SampleList: 후처리된 결과
        """
        batch_size = seg_logits.shape[0]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            # Segmentation 결과
            seg_pred = seg_logits[i].argmax(dim=0, keepdim=True)
            data_samples[i].pred_sem_seg = PixelData(data=seg_pred)

            # Depth 결과
            data_samples[i].pred_depth_map = PixelData(data=depth_pred[i])

        return data_samples
