# Custom Hooks 임포트 (MTL 전용)
custom_imports = dict(
    imports=[
        'hooks.mtl_iter_logger_hook',  # MTL CSV Logger
        'mmseg.engine.hooks.mtl_hooks',  # MTL Hooks
    ],
    allow_failed_imports=False
)

# optimizer
optimizer = dict(type='AdamW', lr=0.0004, weight_decay=0.01)
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed Precision (FP16) 활성화
    accumulative_counts=4,  # grad accumulation: effective batch = per-GPU batch × 4
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0, norm_type=2),  # gradient clipping
    loss_scale='dynamic',  # dynamic loss scaling으로 gradient underflow 방지
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),  # positional block의 weight decay 비활성화
            'norm': dict(decay_mult=0.),  # normalization layer의 weight decay 비활성화
            'head': dict(lr_mult=5.)  # head의 learning rate를 5배로 증가 (transfer learning)
        }))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-2, begin=0, end=500,
        by_epoch=False),
    dict(
        type='PolyLRRatio',
        eta_min_ratio=1e-2,
        power=0.9,
        begin=500,
        end=30000,
        by_epoch=False),
]
# training schedule for 50k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=-1,  # interval 체크포인트 저장 비활성화 (best만 저장)
        # MTL: Seg mIoU best + Depth abs_rel best 둘 다 저장
        save_best=['seg/mIoU', 'depth/abs_rel'],
        rule=['greater', 'less'],  # mIoU는 클수록, abs_rel은 작을수록 좋음
        save_last=True,  # 마지막 checkpoint도 저장 (resume 용)
        max_keep_ckpts=5  # best_mIoU + best_abs_rel + last 유지
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'))
    # MTL 시각화 - 완전 비활성화 (키 자체 제거 + custom_hooks에서도 주석 처리)

# Early Stopping Hook 설정 (MMEngine 내장 버전 사용)
# rule='less': abs_rel은 작을수록 좋으므로 'less' 사용
# patience=20: 20번의 validation 동안 개선 없으면 중단
#             (val_interval=200 기준으로 200 iteration마다 validation 실행, 
#              따라서 최대 4,000 iterations = 20 × 200 더 기다림)
#
# NOTE: MTL에서는 Depth abs_rel을 primary metric으로 사용
#       Seg는 이미 성능이 잘 나오므로 Depth 기준으로 Early Stopping
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='depth/abs_rel',  # MTL primary metric: Depth abs_rel
        rule='less',  # abs_rel은 작을수록 좋음
        min_delta=0.0001,  # 0.5% 개선 (Depth는 더 민감하게)
        patience=20,
        check_finite=True,
    ),
    # MTL Weight Logger Hook - 가중치 변화 추적 및 시각화 (DWA/Uncertainty)
    dict(
        type='MTLWeightLoggerHook',
        log_interval=100
    ),
    # MTL Visualization Hook - 완전 비활성화 (주석 처리)
    # 시각화가 필요한 경우: 학습 후 tools/visualize_mtl_results.py 사용
    # dict(
    #     type='MTLVisualizationHook',
    #     draw=True,
    #     interval=1000,
    #     max_samples=4,
    #     show_seg=True,
    #     show_depth=True,
    #     show_gt=True
    # ),
    # MTL CSV Logger (학습 곡선을 CSV 파일로 저장)
    # Seg mIoU, Depth abs_rel, MTL 가중치 등 모두 기록
    dict(
        type='MTLIterLoggerHook',
        out_csv=None,  # None이면 자동으로 {work_dir}/learning_curve.csv로 저장
        flush_secs=10
    )
]
