# mtl_deploy_cfg.py
ir_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=14,
    save_in_single_file=True,
    input_names=['input'],
    # 중요: 사용자님 모델의 forward 리턴 순서에 맞춰 이름을 지어줍니다.
    output_names=['seg_logits', 'depth_map'], 
    input_shape=[512, 512],
    save_file='end2end.onnx')

backend_config = dict(
    type='tensorrt', # 최종 목적지가 젯슨이므로 TensorRT 설정
    common_config=dict(
        fp16_mode=True, # 젯슨 오린의 Tensor Core 활용을 위해 필수
        max_workspace_size=1 << 30),
        model_inputs=[
        dict(
            from_step='onnx2tensorrt',
            to_step='tensorrt',
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 512],
                    opt_shape=[1, 3, 512, 512],
                    max_shape=[1, 3, 512, 512])))
    ])

codebase_config = dict(
    type='mmseg', # mmseg 기반임을 명시
    task='Segmentation',
    with_argmax=False) # 추론 시 후처리를 직접 통제하기 위해 False 권장