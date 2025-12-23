import torch
from typing import Dict, Union, Tuple
from mmseg.registry import MODELS
from mmseg.utils import SampleList, add_prefix
from .mtl_encoder_decoder import MTLEncoderDecoder
from mmseg.models.utils import resize



def patched_unflatten(self, dim, sizes):
    # 1. 차원 계산
    dims = self.dim()
    actual_dim = dim if dim >= 0 else dim + dims
    
    # 2. 새로운 Shape 계산
    old_shape = list(self.shape)
    new_shape = old_shape[:actual_dim] + list(sizes) + old_shape[actual_dim + 1:]
    
    # [핵심 수정] .contiguous()를 추가하여 메모리 충돌 방지
    return self.contiguous().view(*new_shape)

# 패치 적용 (이 줄이 실행되어야 torch 기능이 바뀐다)
torch.Tensor.unflatten = patched_unflatten
# ==========================================

@MODELS.register_module()
class MTLInteractionSegmentor(MTLEncoderDecoder):
    """Segmentation 특징을 Depth Head로 전달하는 상호작용형 MTL 모델"""

    def __init__(self, 
                 backbone: Dict, 
                 seg_decode_head: Dict, 
                 depth_decode_head: Dict, 
                 **kwargs): # <-- MMDeploy가 보내는 'decode_head'를 흡수하여 에러 방지
        """
        초기화 시 **kwargs를 사용하여 설정 파일의 유연한 확장을 지원합니다.
        """
        """
        MMDeploy에서 강제로 주입한 'decode_head'를 제거한 후 부모 클래스를 초기화합니다.
        """
        # [핵심] 부모 클래스에게 넘겨주기 전에 'decode_head'가 있다면 삭제합니다.
        kwargs.pop('decode_head', None)
        super().__init__(
            backbone=backbone,
            seg_decode_head=seg_decode_head,
            depth_decode_head=depth_decode_head,
            **kwargs
        )

    def forward(self, inputs, *args, mode='tensor', **kwargs):
        if mode == 'tensor' or torch.jit.is_tracing():
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            
            x = self.extract_feat(inputs)
            seg_logits, seg_feat = self.seg_decode_head.forward_with_feat(x)
            
            # [중요 수정] inputs.shape[-2:] 대신 고정 해상도 사용
            # config 파일의 input_shape와 일치해야 합니다 (예: 512, 512)
            target_size = (512, 512) 

            seg_logits = resize(
                input=seg_logits,
                size=target_size, # <-- 여기를 고정값으로 변경
                mode='bilinear',
                align_corners=self.seg_decode_head.align_corners)

            depth_pred = self.depth_decode_head.forward(x, seg_feat.detach())
            
            return (seg_logits, depth_pred)

        # 기존 학습/추론 로직
        if mode == 'loss':
            return self.loss(inputs, args[0] if args else kwargs.get('data_samples'))
        elif mode == 'predict':
            return self.predict(inputs, args[0] if args else kwargs.get('data_samples'))

    def loss(self, inputs: torch.Tensor, data_samples: SampleList) -> dict:
        """기존 학습 로직 유지"""
        x = self.extract_feat(inputs)
        losses = dict()

        seg_logits, seg_feat = self.seg_decode_head.forward_with_feat(x)
        seg_loss = self.seg_decode_head.loss_by_feat(seg_logits, data_samples)
        losses.update(add_prefix(seg_loss, 'seg'))

        depth_logits = self.depth_decode_head.forward(x, seg_feat.detach())
        depth_loss = self.depth_decode_head.loss_by_feat(depth_logits, data_samples)
        losses.update(add_prefix(depth_loss, 'depth'))

        if self.with_auxiliary_head:
            loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        self._apply_mtl_weights(losses)
        return losses

    def predict(self, inputs: torch.Tensor, data_samples: SampleList = None) -> SampleList:
        """기존 추론 로직 유지 (eval 시 사용)"""
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict()] * inputs.shape[0]

        x = self.extract_feat(inputs)
        seg_logits, seg_feat = self.seg_decode_head.forward_with_feat(x)
        
        seg_logits = resize(
            input=seg_logits,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.seg_decode_head.align_corners)

        depth_pred = self.depth_decode_head.forward(x, seg_feat.detach())
        
        return self.postprocess_result(seg_logits, depth_pred, data_samples)