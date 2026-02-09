# chamnet_mtl_base: SegFormer MiT-B0 기반 Multi-Task Learning (Segmentation + Depth)
# 목표: CLI arguments로 weight strategy를 유연하게 선택 가능
#
# 사용법:
#   DWA:         python tools/train_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdata.py --weight-strategy dwa
#   Uncertainty: python tools/train_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdata.py --weight-strategy uncertainty
#   Manual:      python tools/train_mtl.py configs/chamnet/chamnet_mtl_base_segformer_chamdata.py --weight-strategy manual --seg-weight 1.0 --depth-weight 0.5

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_30m.py',  # MTL 전용 schedule
    '../_base_/datasets/chamdata_mtl_met.py'     # MTL 전용 dataset
]

# Custom MTL hooks 및 모듈 import
custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.mtl_encoder_decoder',
        'mmseg.models.decode_heads.depth_head',  # InteractionDepthHead가 이 파일 안에 있음
        'mmseg.models.losses.depth_losses',
        'mmseg.datasets.mtl_dataset',
        'mmseg.evaluation.metrics.depth_metric',
        'mmseg.models.data_preprocessor',
        'mmseg.engine.hooks.mtl_hooks',
        'hooks',
        # InteractionSegformerHead도 별도 파일이 아니라 segformer_head.py에 넣으셨다면 아래처럼 수정
        'mmseg.models.decode_heads.interaction_segformer_head', 
        'mmseg.models.segmentors.mtl_interaction_segmentor'
    ],
    allow_failed_imports=False
)
# SegFormer MiT-B0 pretrained checkpoint
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

# ============================================================================
# MTL 학습 설정 (기본값, CLI arguments로 override 가능)
# ============================================================================
mtl_config = dict(
    # Weight strategy (CLI로 override됨)
    # 'dwa', 'uncertainty', 'manual'/'fixed' 중 선택
    weight_strategy='manual',  # 기본값

    # DWA (Dynamic Weight Average) 설정
    dwa_config=dict(
        enabled=False,  # CLI로 자동 활성화됨
        num_tasks=2,    # Segmentation + Depth
        temperature=2.0,
        window_size=10,
        update_freq=50,  # 몇 iteration마다 가중치 업데이트
    ),

    # Uncertainty Weighting 설정
    uncertainty_config=dict(
        enabled=True,  # Uncertainty weighting 활성화 (체크포인트 로딩 시 필요)
        init_log_var_seg=0.0,
        init_log_var_depth=0.0,
    ),

    # Fixed/Manual weights (기본값)
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

    # ===== Shared Backbone: SegFormer MiT-B0 =====
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,  # B0 variant
        num_stages=4,
        num_layers=[2, 2, 2, 2],
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

    # ===== Segmentation Head =====
    seg_decode_head=dict(
        type='InteractionSegformerHead',
        in_channels=[32, 64, 160, 256],  # MiT-B0 output channels
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
    in_channels=[32, 64, 160, 256],
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
    dict(type='GradientLoss', loss_weight=0.5,  min_depth=0.1)
    
]

),
    # MTL 설정 전달
    mtl_config=mtl_config,

    # 학습/테스트 설정
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# 추가 설정
# ============================================================================
work_dir = './work_dirs/chamnet_segformer_dwa_l1+GL'  # CLI로 자동 override됨

# Visualization 완전 비활성화 (키 자체 제거)
# - 학습 속도 향상 + 디스크 공간 절약 + mmseg 의존성 오류 방지
# - 시각화가 필요한 경우: python tools/visualize_mtl_results.py <config> <checkpoint>
# default_hooks에서 visualization 키 자체를 제거 (None은 AssertionError 발생)

load_from = None
resume = False

'''# 실험 1: 현재 (베이스라인)
loss_decode=[
    dict(type='L1Loss', loss_weight=1.0, min_depth=0.1),
]

# 실험 2: L1 + BerHu
loss_decode=[
    dict(type='L1Loss', loss_weight=1.0, min_depth=0.1),
    dict(type='BerHuLoss', loss_weight=0.3, threshold=0.2, min_depth=0.1),
]

# 실험 3: L1 + Gradient
loss_decode=[
    dict(type='L1Loss', loss_weight=1.0, min_depth=0.1),
    dict(type='GradientLoss', loss_weight=0.5, min_depth=0.1),
]

# 실험 4: L1 + BerHu + Gradient (종합)
loss_decode=[
    dict(type='L1Loss', loss_weight=1.0, min_depth=0.1),
    dict(type='BerHuLoss', loss_weight=0.3, threshold=0.2, min_depth=0.1),
    dict(type='GradientLoss', loss_weight=0.5, min_depth=0.1),
]'''

''' Loss 조합에 따른 차이는 미미하였음 ```