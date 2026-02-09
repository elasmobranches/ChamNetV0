# chamnet_mtl_segmenter: Segmenter(ViT) 기반 Multi-Task Learning (Segmentation + Depth)
# Pure transformer architecture for semantic segmentation
#
# 사용법:
#   python tools/train_mtl.py configs/chamnet/chamnet_mtl_segmenter_metric.py --weight-strategy uncertainty

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_30m.py',
    '../_base_/datasets/chamdata_mtl_met.py'
]

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

# ViT-Base pretrained checkpoint
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'

# ============================================================================
# MTL 학습 설정
# ============================================================================
mtl_config = dict(
    weight_strategy='uncertainty',
    dwa_config=dict(
        enabled=False,
        num_tasks=2,
        temperature=2.0,
        window_size=10,
        update_freq=50,
    ),
    uncertainty_config=dict(
        enabled=True,
        init_log_var_seg=0.0,
        init_log_var_depth=0.0,
    ),
    fixed_weights=dict(
        seg_weight=1.0,
        depth_weight=5.0,
    ),
)

# ============================================================================
# 모델 정의
# ============================================================================
norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)  # ViT uses LayerNorm

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

    # ===== Backbone: Vision Transformer (ViT-Base) =====
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,  # ViT-Base
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),  # Extract features from multiple layers
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),

    # ===== Segmentation Head =====
    # Note: ViT outputs uniform dimension (768) from all layers
    seg_decode_head=dict(
        type='InteractionSegformerHead',
        in_channels=[768, 768, 768, 768],  # ViT-Base uniform output
        in_index=[0, 1, 2, 3],
        channels=256,  # Increased channels for ViT
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # Head uses BN
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

    # ===== Depth Head =====
    depth_decode_head=dict(
        type='InteractionDepthHead',
        in_channels=[768, 768, 768, 768],  # Same as seg head
        in_index=[0, 1, 2, 3],
        channels=256,  # Increased channels for ViT
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # Head uses BN
        align_corners=False,
        min_depth=0.1,
        max_depth=10.0,
        loss_decode=[
            dict(type='L1Loss', loss_weight=1.0, min_depth=0.1),
            dict(type='GradientLoss', loss_weight=0.5, min_depth=0.1)
        ]
    ),

    mtl_config=mtl_config,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# 추가 설정
# ============================================================================
work_dir = './work_dirs/chamnet_segmenter_mtl'
load_from = None
resume = False
