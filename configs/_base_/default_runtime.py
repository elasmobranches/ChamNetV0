default_scope = 'mmseg'

# 재현성을 위한 random seed 설정
randomness = dict(
    seed=42,              # 고정할 seed 값
    deterministic=False,  # True로 하면 완벽한 재현성 (하지만 느려짐)
    diff_rank_seed=False  # 분산 학습 시 각 rank마다 다른 seed
)

env_cfg = dict(
    cudnn_benchmark=True,  # 재현성을 위해 False로 하면 더 확실하지만 느려짐
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'  # 학습 진행 상황 확인을 위해 INFO로 설정
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
