# deploy_model_cfg.py

_base_ = ['./configs/chamnet/chamnet_mtl_base_segformer_chamdatav3.py']

model = dict(
    decode_head=dict(type='InteractionSegformerHead') 
)

# mmseg 표준을 준수하면서도 가장 가벼운 파이프라인
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    # PackSegInputs는 mmseg 기본 라이브러리에 있으므로 KeyError가 나지 않습니다.
    # meta_keys를 비워둠으로써 Tracer가 싫어하는 파이썬 객체 정보를 최소화합니다.
    dict(type='PackSegInputs', meta_keys=[]) 
]

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)