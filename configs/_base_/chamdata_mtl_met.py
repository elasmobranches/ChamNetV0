# chamdata_mtl.py
# MDE + Segmentation Multi-Task Learning 데이터셋 설정 파일

# 1. 기본 설정
dataset_type = 'MTLChamDataset'
data_root = 'dataset'

# ==============================================================================
# 2. 파이프라인 설정 (핵심 수정: Normalize 제거)
# ==============================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MTLLoadDepthAnnotation'), # .npy 파일 로드 (미터 단위 그대로 유지)
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    
    # [수정됨] Metric Depth 학습을 위해 Normalize 제거
    # 이 줄이 없어야 모델이 "0.5"가 아닌 "5.0m"를 배웁니다.
    # dict(type='MTLNormalizeDepth', max_depth=10.0), 
    
    dict(type='PackMTLSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MTLLoadDepthAnnotation'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    
    # [수정됨] 검증 때도 미터 단위 그대로 비교해야 하므로 제거
    # dict(type='MTLNormalizeDepth', max_depth=10.0),
    
    dict(type='PackMTLSegInputs'),
]

test_pipeline = val_pipeline

# ==============================================================================
# 3. 데이터로더 설정
# ==============================================================================
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 학습 데이터 확장자 설정 (사용자 환경 기준)
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
        # 검증 데이터 확장자 설정 (User: .jpg)
        img_suffix='.jpg',
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
        # 테스트 데이터 확장자 설정 (User: .jpg)
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

# ==============================================================================
# 4. 평가 메트릭 설정 (유효 픽셀 필터링)
# ==============================================================================
val_evaluator = [
    # Segmentation 평가
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='seg'),
    
    # Depth 평가
    dict(
        type='DepthMetric',
        depth_metrics=['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3'],

        # [핵심] 학습 범위(0.1~10.0m)와 평가 범위를 일치시킴 (정합성 확보)
        # - min_depth_eval=0.1: 학습하지 않은 Dead Zone(0.001~0.1m) 제외
        # - max_depth_eval=10.0: 센서 노이즈(10m 초과) 제외
        # - 실제 데이터: 0~0.1m 구간은 전체의 0.04%로 영향 미미
        min_depth_eval=0.1,
        max_depth_eval=10.0,

        prefix='depth'
    )
]

test_evaluator = val_evaluator