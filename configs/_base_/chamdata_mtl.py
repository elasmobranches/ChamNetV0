# dataset settings - Multi-Task Learning (Segmentation + Depth)
dataset_type = 'MTLChamDataset'
data_root = 'dataset'

# MTL 파이프라인 수정
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 1. LoadDepthAnnotation -> MTLLoadDepthAnnotation으로 변경
    dict(type='MTLLoadDepthAnnotation'), 
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # 2. 모델이 학습하기 좋게 0~1 사이로 깎아주는 단계 추가 (필수!)
    dict(type='MTLNormalizeDepth', max_depth=10.0), 
    dict(type='PackMTLSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 여기도 동일하게 MTLLoadDepthAnnotation
    dict(type='MTLLoadDepthAnnotation'), 
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # 여기도 동일하게 MTLNormalizeDepth
    dict(type='MTLNormalizeDepth', max_depth=10.0), 
    dict(type='PackMTLSegInputs'),
]

test_pipeline = val_pipeline

# --- dataloaders ---
train_dataloader = dict(
    batch_size=4,  # MTL은 메모리를 더 사용
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        depth_map_suffix='_depth.npy',
        serialize_data=False,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/masks',
            depth_map_path='train/metric_depth/depth_npy',
        ),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',  # Valid는 jpg
        seg_map_suffix='_mask.png',
        depth_map_suffix='_depth.npy',
        serialize_data=False,
        data_prefix=dict(
            img_path='valid/images',
            seg_map_path='valid/masks',
            depth_map_path='valid/metric_depth/depth_npy',
        ),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='_mask.png',
        depth_map_suffix='_depth.npy',
        serialize_data=False,
        data_prefix=dict(
            img_path='test/images',
            seg_map_path='test/masks',
            depth_map_path='test/metric_depth/depth_npy',
        ),
        pipeline=test_pipeline))

# MTL 평가 메트릭
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
    dict(
        type='DepthMetric',
        depth_metrics=['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3'],
        min_depth_eval=0.001,
        max_depth_eval=10.0,
        prefix='depth'
    )
]

test_evaluator = val_evaluator

