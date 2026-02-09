# chamnet_mtl_mobilenetv3: MobileNetV3 기반 Multi-Task Learning (Segmentation + Depth)
# 경량 모델로 빠른 학습 및 추론 가능
#
# 사용법:
#   python tools/train_mtl.py configs/chamnet/chamnet_mtl_mobilenetv3_metric.py --weight-strategy uncertainty

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

# MobileNetV3 pretrained checkpoint
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth'

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

    # ===== Backbone: MobileNetV3 Small =====
    # MobileNetV3 Small layer output channels:
    # layer1(idx=1)=16, layer3(idx=3)=24, layer6(idx=6)=40, layer11(idx=11)=96
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(1, 3, 6, 11),  # 4 stage outputs matching in_channels=[16, 24, 40, 96]
        norm_cfg=norm_cfg),

    # ===== Segmentation Head =====
    seg_decode_head=dict(
        type='InteractionSegformerHead',
        in_channels=[16, 24, 40, 96],  # MobileNetV3 Small output channels
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

    # ===== Depth Head =====
    depth_decode_head=dict(
        type='InteractionDepthHead',
        in_channels=[16, 24, 40, 96],  # Same as seg head (MobileNetV3 Small)
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
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
work_dir = './work_dirs/chamnet_mobilenetv3_mtl'
load_from = None
resume = False
