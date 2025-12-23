# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

@MODELS.register_module()
class InteractionSegformerHead(BaseDecodeHead):
    """SegformerHead with interaction support for MTL.
    
    This head extends the original SegformerHead to support feature interaction
    between segmentation and depth tasks in Multi-Task Learning.
    """
    
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        
        assert num_inputs == len(self.in_index)
        
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def _forward_feature(self, inputs):
        """기본 forward 로직 중 분류(cls_seg) 전까지의 특징 맵 추출 공통 로직"""
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # 모든 스테이지 특징을 cat하고 fusion_conv를 통과시킨 결과 (B, C, H/4, W/4)
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def forward(self, inputs):
        """기존 forward: 특징 추출 후 바로 분류 결과 반환"""
        out = self._forward_feature(inputs)
        out = self.cls_seg(out)
        return out

    def forward_with_feat(self, inputs):
        """새로 추가: 분류 결과(logits)와 퓨전된 특징 맵(feat)을 함께 반환"""
        feat = self._forward_feature(inputs)
        logits = self.cls_seg(feat)
        return logits, feat
