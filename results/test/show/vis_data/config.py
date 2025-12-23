checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False,
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
        'mmseg.models.segmentors.mtl_interaction_segmentor',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='MTLSegDataPreProcessor')
data_root = 'dataset'
dataset_type = 'MTLChamDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=-1,
        max_keep_ckpts=5,
        rule=[
            'greater',
            'less',
        ],
        save_best=[
            'seg/mIoU',
            'depth/abs_rel',
        ],
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=200, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/workspace/mmsegmentation/work_dirs/chamnet_mtl_manual_1_5_last_ch128_int/best_depth_abs_rel_iter_21400.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='MTLSegDataPreProcessor'),
    depth_decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                lambda_variance=0.85,
                loss_name='loss_depth_silog',
                loss_weight=1.0,
                type='SILogLoss'),
            dict(
                loss_name='loss_depth_berhu',
                loss_weight=0.3,
                threshold=0.2,
                type='BerHuLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        type='InteractionDepthHead'),
    mtl_config=dict(
        dwa_config=dict(
            enabled=False,
            num_tasks=2,
            temperature=2.0,
            update_freq=50,
            window_size=10),
        fixed_weights=dict(depth_weight=5.0, seg_weight=1.0),
        uncertainty_config=dict(
            enabled=True, init_log_var_depth=0.0, init_log_var_seg=0.0),
        weight_strategy='manual'),
    pretrained=None,
    seg_decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                class_weight=[
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                eps=0.001, loss_weight=0.5, type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        type='InteractionSegformerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='MTLInteractionSegmentor')
mtl_config = dict(
    dwa_config=dict(
        enabled=False,
        num_tasks=2,
        temperature=2.0,
        update_freq=50,
        window_size=10),
    fixed_weights=dict(depth_weight=5.0, seg_weight=1.0),
    uncertainty_config=dict(
        enabled=True, init_log_var_depth=0.0, init_log_var_seg=0.0),
    weight_strategy='manual')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=4,
    clip_grad=dict(max_norm=5.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0004, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=5.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.0004, type='AdamW', weight_decay=0.01)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.01, type='LinearLR'),
    dict(
        begin=500,
        by_epoch=False,
        end=30000,
        eta_min_ratio=0.01,
        power=0.9,
        type='PolyLRRatio'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='test/metric_depth/depth_npy',
            img_path='test/images',
            seg_map_path='test/masks'),
        data_root='dataset',
        depth_map_suffix='_depth.npy',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='MTLLoadDepthAnnotation'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(max_depth=10.0, type='MTLNormalizeDepth'),
            dict(type='PackMTLSegInputs'),
        ],
        seg_map_suffix='_mask.png',
        serialize_data=False,
        type='MTLChamDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        iou_metrics=[
            'mIoU',
        ],
        keep_results=True,
        output_dir=
        './work_dirs/chamnet_mtl_manual_1_5_last_ch128_int/test/preds',
        prefix='seg',
        type='IoUMetric'),
    dict(
        depth_metrics=[
            'abs_rel',
            'sq_rel',
            'rmse',
            'rmse_log',
            'd1',
            'd2',
            'd3',
        ],
        keep_results=True,
        max_depth_eval=10.0,
        min_depth_eval=0.001,
        output_dir=
        './work_dirs/chamnet_mtl_manual_1_5_last_ch128_int/test/preds',
        prefix='depth',
        type='DepthMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MTLLoadDepthAnnotation'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(max_depth=10.0, type='MTLNormalizeDepth'),
    dict(type='PackMTLSegInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=200)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='train/metric_depth/depth_npy',
            img_path='train/images',
            seg_map_path='train/masks'),
        data_root='dataset',
        depth_map_suffix='_depth.npy',
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='MTLLoadDepthAnnotation'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(max_depth=10.0, type='MTLNormalizeDepth'),
            dict(type='PackMTLSegInputs'),
        ],
        seg_map_suffix='_mask.png',
        serialize_data=False,
        type='MTLChamDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MTLLoadDepthAnnotation'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(max_depth=10.0, type='MTLNormalizeDepth'),
    dict(type='PackMTLSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='valid/metric_depth/depth_npy',
            img_path='valid/images',
            seg_map_path='valid/masks'),
        data_root='dataset',
        depth_map_suffix='_depth.npy',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='MTLLoadDepthAnnotation'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(max_depth=10.0, type='MTLNormalizeDepth'),
            dict(type='PackMTLSegInputs'),
        ],
        seg_map_suffix='_mask.png',
        serialize_data=False,
        type='MTLChamDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(iou_metrics=[
        'mIoU',
    ], prefix='seg', type='IoUMetric'),
    dict(
        depth_metrics=[
            'abs_rel',
            'sq_rel',
            'rmse',
            'rmse_log',
            'd1',
            'd2',
            'd3',
        ],
        max_depth_eval=10.0,
        min_depth_eval=0.001,
        prefix='depth',
        type='DepthMetric'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MTLLoadDepthAnnotation'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(max_depth=10.0, type='MTLNormalizeDepth'),
    dict(type='PackMTLSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='./work_dirs/chamnet_mtl_manual_1_5_last_ch128_int/test/show',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/chamnet_mtl_manual_1_5_last_ch128_int/test'
