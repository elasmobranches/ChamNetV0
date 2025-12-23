"""
Depth Prediction Head for MTL

SegFormer와 호환되는 Depth 예측 헤드를 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize


@MODELS.register_module()
class DepthHead(SegformerHead):
    """Depth Prediction Head (SegFormer 구조 기반)
    
    SegFormer의 All-MLP 디코더 구조를 활용하여 Depth 예측을 수행합니다.
    
    Args:
        in_channels (List[int]): 입력 채널 수 리스트
        in_index (List[int]): 사용할 feature map 인덱스
        channels (int): 디코더 채널 수
        dropout_ratio (float): Dropout 비율
        num_classes (int): 출력 채널 수 (Depth는 1)
        norm_cfg (dict): Normalization 설정
        align_corners (bool): Resize 시 align_corners 옵션
        loss_decode (dict): 손실 함수 설정
    """
    
    def __init__(self,
                 in_channels,
                 in_index,
                 channels,
                 dropout_ratio=0.1,
                 num_classes=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 align_corners=False,
                 loss_decode=dict(
                     type='SILogLoss',
                     loss_weight=1.0,
                     loss_name='loss_depth'),
                 **kwargs):
        
        # Depth head는 binary segmentation이 아니므로 threshold 불필요
        # 하지만 경고를 피하기 위해 임의의 값(0.5)으로 설정
        kwargs['threshold'] = 0.5
        
        # SegformerHead 초기화 (num_classes=1로 설정)
        super().__init__(
            in_channels=in_channels,
            in_index=in_index,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=1,  # Depth는 항상 1채널
            norm_cfg=norm_cfg,
            align_corners=align_corners,
            loss_decode=loss_decode,
            **kwargs
        )
        
        # Depth 전용 출력 활성화
        self.depth_activation = nn.Sigmoid()
        
        # conv_seg bias를 작은 음수로 초기화하여 Sigmoid 포화 방지
        # Sigmoid(0) = 0.5이므로 bias를 -2로 하면 Sigmoid(-2) ≈ 0.12
        if hasattr(self, 'conv_seg'):
            nn.init.constant_(self.conv_seg.bias, -2.0)
    
    def forward(self, inputs):
        """Forward 함수
        
        Args:
            inputs: Multi-level features
        
        Returns:
            Tensor: Depth predictions [B, 1, H, W]
        """
        # SegFormer의 forward 사용 (1/4 크기 출력)
        output = super().forward(inputs)
        
        # 512x512로 고정 upsampling (training & validation 일관성 유지)
        target_size = (512, 512)
        output = F.interpolate(
            output, 
            size=target_size, 
            mode='bilinear', 
            align_corners=self.align_corners
        )
        
        # Depth는 양수여야 하므로 Sigmoid 적용
        # SILogLoss는 양수 값이 필요하므로 학습 시에도 적용
        output = self.depth_activation(output)
        
        return output
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """손실 계산
        
        Args:
            seg_logits: 모델 출력 (실제로는 depth predictions)
            batch_data_samples: GT depth 포함된 데이터
        
        Returns:
            dict: 손실 딕셔너리
        """
        # Depth GT 추출
        depth_targets = self._stack_batch_depth(batch_data_samples)
        
        # 손실 계산
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    depth_targets,
                    weight=None,
                    ignore_index=None)
            else:
                losses[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    depth_targets,
                    weight=None,
                    ignore_index=None)
        
        return losses
    
    def _stack_batch_depth(self, batch_data_samples):
        """배치에서 depth GT 추출 및 스택
        
        Args:
            batch_data_samples: 데이터 샘플 리스트
        
        Returns:
            Tensor: Stacked depth maps
        """
        depth_targets = []
        for data_sample in batch_data_samples:
            # SegDataSample 객체에서 depth 추출
            if hasattr(data_sample, 'gt_depth_map'):
                depth = data_sample.gt_depth_map.data
            elif hasattr(data_sample, 'gt_depth'):
                depth = data_sample.gt_depth.data
            else:
                # Depth GT가 없으면 0으로 채운 텐서 생성
                import warnings
                warnings.warn("No depth GT found in data_sample, using zeros")
                gt_sem_seg_shape = data_sample.gt_sem_seg.data.shape
                depth = torch.zeros(1, gt_sem_seg_shape[-2], gt_sem_seg_shape[-1],
                                  device=data_sample.gt_sem_seg.data.device)
            
            depth_targets.append(depth)
        
        return torch.cat(depth_targets, dim=0)


@MODELS.register_module()
class LightweightDepthHead(BaseDecodeHead):
    """경량 Depth Head (더 간단한 구조)
    
    메모리가 제한적일 때 사용할 수 있는 간단한 Depth 헤드입니다.
    
    Args:
        in_channels (int or List[int]): 입력 채널 수
        channels (int): 중간 채널 수
        num_convs (int): Convolution 레이어 수
        concat_input (bool): 입력 concat 여부
        dropout_ratio (float): Dropout 비율
        num_classes (int): 출력 채널 수 (Depth는 1)
        norm_cfg (dict): Normalization 설정
        align_corners (bool): Resize 시 align_corners 옵션
        loss_decode (dict): 손실 함수 설정
    """
    
    def __init__(self,
                 in_channels,
                 channels=256,
                 num_convs=2,
                 concat_input=False,
                 dropout_ratio=0.1,
                 num_classes=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 align_corners=False,
                 loss_decode=dict(
                     type='SILogLoss',
                     loss_weight=1.0,
                     loss_name='loss_depth'),
                 **kwargs):
        
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=1,  # Depth는 항상 1채널
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            align_corners=align_corners,
            loss_decode=loss_decode,
            **kwargs
        )
        
        self.concat_input = concat_input
        self.num_convs = num_convs
        
        # Convolution layers
        convs = []
        for i in range(num_convs):
            in_ch = self.in_channels if i == 0 else channels
            convs.append(
                ConvModule(
                    in_ch,
                    channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU'))
            )
        self.convs = nn.Sequential(*convs)
        
        # Concat 사용 시 채널 조정
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + channels,
                channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
        
        # Dropout
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        # 최종 예측 레이어
        self.conv_depth = nn.Conv2d(channels, 1, 1)
        
        # Depth 활성화
        self.depth_activation = nn.Sigmoid()
    
    def forward(self, inputs):
        """Forward 함수
        
        Args:
            inputs: Feature map 또는 multi-level features
        
        Returns:
            Tensor: Depth predictions [B, 1, H, W]
        """
        # 단일 레벨 또는 멀티 레벨 feature 처리
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[self.in_index]
        
        x = inputs
        output = self.convs(x)
        
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        output = self.conv_depth(output)
        
        # Inference 시 Sigmoid 적용
        if not self.training:
            output = self.depth_activation(output)
        
        return output
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """손실 계산
        
        Args:
            seg_logits: 모델 출력 (depth predictions)
            batch_data_samples: GT depth 포함된 데이터
        
        Returns:
            dict: 손실 딕셔너리
        """
        # Depth GT 추출
        depth_targets = []
        for data_sample in batch_data_samples:
            if 'gt_depth' in data_sample:
                depth = data_sample['gt_depth']['data']
            elif 'gt_depth_map' in data_sample:
                depth = data_sample['gt_depth_map']['data']
            else:
                # 경고 후 zero tensor
                import warnings
                warnings.warn("No depth GT found, using zeros")
                depth = torch.zeros_like(seg_logits[0:1])
            
            depth_targets.append(depth)
        
        depth_targets = torch.stack(depth_targets, dim=0)
        
        # 크기 맞추기
        if seg_logits.shape[2:] != depth_targets.shape[2:]:
            depth_targets = resize(
                depth_targets,
                size=seg_logits.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
        
        # 손실 계산
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss_decode(
                    seg_logits.squeeze(1) if seg_logits.shape[1] == 1 else seg_logits,
                    depth_targets.squeeze(1) if depth_targets.shape[1] == 1 else depth_targets
                )
            else:
                losses[loss_decode.loss_name] += loss_decode(
                    seg_logits.squeeze(1) if seg_logits.shape[1] == 1 else seg_logits,
                    depth_targets.squeeze(1) if depth_targets.shape[1] == 1 else depth_targets
                )
        
        return losses

@MODELS.register_module()
class InteractionDepthHead(DepthHead):
    """Segmentation 특징을 Residual Gated Fusion으로 결합하는 지능형 Depth Head"""
    
    def __init__(self, interaction_channels=None, **kwargs):
        super().__init__(**kwargs)
        
        fusion_in_channels = self.channels * 2
        fusion_out_channels = interaction_channels or self.channels
        
        # 1. 정보를 통합할 Fusion Conv
        self.interaction_fusion = ConvModule(
            in_channels=fusion_in_channels,
            out_channels=fusion_out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # 2. 어느 정보를 얼마나 가져올지 결정하는 Gate
        self.gate = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, seg_feat=None):
        # 1. Depth 특징 추출 (SegformerHead 로직 활용)
        x = self._transform_inputs(inputs)
        outs = []
        for i in range(len(x)):
            layer_out = self.convs[i](x[i])
            outs.append(
                resize(
                    input=layer_out,
                    size=x[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )
        depth_feat = self.fusion_conv(torch.cat(outs, dim=1))
        output_size = depth_feat.shape[-2:] # 특징 맵의 현재 해상도 (예: 128x128)

        # 2. Seg 특징 결합 (Residual Gated Fusion)
        if seg_feat is not None:
            # 해상도 동적 일치
            if seg_feat.shape[-2:] != output_size:
                seg_feat = F.interpolate(
                    seg_feat, size=output_size, mode='bilinear', align_corners=self.align_corners)
            
            combined = torch.cat([depth_feat, seg_feat], dim=1)
            gate_value = self.gate(combined)
            fused_info = self.interaction_fusion(combined)
            
            # [피드백 반영] Residual Gated Update: $depth\_feat + gate \times (fused\_info - depth\_feat)$
            depth_feat = depth_feat + gate_value * (fused_info - depth_feat)

        # 3. 최종 Depth 예측 (1채널 로짓 생성)
        out = self.cls_seg(depth_feat)

        # 4. [중요] 동적 업샘플링 (하드코딩 제거)
        # 상위 모듈이 처리해주길 기다리는 대신, 입력된 특징의 4배(백본의 1/4 크기인 inputs[0])를 
        # 기준으로 원본 해상도를 추론하여 자동으로 키워줍니다.
        target_h, target_w = inputs[0].shape[-2] * 4, inputs[0].shape[-1] * 4
        out = resize(
            input=out,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=self.align_corners
        )

        # 5. 활성화 함수
        if hasattr(self, 'depth_activation') and self.depth_activation is not None:
            out = self.depth_activation(out)
            
        return out