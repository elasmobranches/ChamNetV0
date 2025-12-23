# chamnet_mtl_b2_interaction: SegFormer MiT-B2 기반 Multi-Task Learning
# 수정 사항: Backbone 구조 확장 및 Head 채널 정합성 확보

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_30m.py',  # MTL 전용 schedule
    '../_base_/datasets/chamdata_mtl.py'     # MTL 전용 dataset
]

# Custom MTL 모듈 import
custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.mtl_encoder_decoder',
        'mmseg.models.decode_heads.depth_head',
        'mmseg.models.losses.depth_losses',
        'mmseg.datasets.mtl_dataset',
        'mmseg.evaluation.metrics.depth_metric',
        'mmseg.models.data_preprocessor',
        'mmseg.engine.hooks.mtl_hooks',
        'hooks',
        'mmseg.models.decode_heads.interaction_segformer_head', 
        'mmseg.models.segmentors.mtl_interaction_segmentor'
    ],
    allow_failed_imports=False
)

# [수정] SegFormer MiT-B2 pretrained checkpoint
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

# ============================================================================
# MTL 학습 설정
# ============================================================================
mtl_config = dict(
    weight_strategy='manual', 
    uncertainty_config=dict(
        enabled=True,
        init_log_var_seg=0.0,
        init_log_var_depth=0.0,
    ),
    fixed_weights=dict(
        seg_weight=1.0,
        depth_weight=5.0, # Depth 가중치를 높여 밸런스 유지
    ),
)

# ============================================================================
# 모델 정의
# ============================================================================
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='MTLSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))

model = dict(
    type='MTLInteractionSegmentor',
    data_preprocessor=data_preprocessor,
    pretrained=None,

    # [수정] Shared Backbone: SegFormer MiT-B2 사양
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,           # B0(32) -> B2(64)
        num_stages=4,
        num_layers=[3, 4, 6, 3],  # B0([2,2,2,2]) -> B2([3,4,6,3])
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),

    # [수정] Segmentation Head (B2 출력 채널에 맞춤)
    seg_decode_head=dict(
        type='InteractionSegformerHead',
        in_channels=[64, 128, 320, 512],  # B2 사양으로 변경
        in_index=[0, 1, 2, 3],
        channels=128,                    
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=0.5,
                eps=1e-3)
        ]),

    # [수정] Depth Head (B2 출력 채널 및 Interaction 로직)
    depth_decode_head=dict(
        type='InteractionDepthHead',
        in_channels=[64, 128, 320, 512], # B2 사양으로 변경
        in_index=[0, 1, 2, 3],
        channels=128,                   # Seg Head와 동일하게 256 확장
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='SILogLoss',
                loss_weight=1.0,
                loss_name='loss_depth_silog',
            ),
            dict(
                type='BerHuLoss',
                loss_weight=0.3,
                loss_name='loss_depth_berhu',
                threshold=0.2,
            )
        ]
    ),
    mtl_config=mtl_config,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# 추가 설정
# ============================================================================
work_dir = './work_dirs/chamnet_mtl_manual_b2_interaction_128' # 경로명 명확히 수정

load_from = None
resume = False