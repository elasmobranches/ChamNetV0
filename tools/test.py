# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='show prediction results (default: off)')
    parser.add_argument(
        '--show-dir',
        default=None,
        help='directory where painted images will be saved. '
        'If not specified, defaults to work_dir/test/show')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta',
        action='store_true',
        default=False,
        help='Test time augmentation (default: off)')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Handle None case (visualization disabled via cfg_options)
        if visualization_hook is None:
            # Skip visualization if explicitly disabled
            return cfg
        # Turn on visualization (only for hooks that support 'draw' parameter)
        # SegVisualizationHook supports 'draw', but DepthVisualizationHook does not
        hook_type = visualization_hook.get('type', '')
        if 'Seg' in hook_type or 'MTL' in hook_type:
            visualization_hook['draw'] = True
        # For Depth hooks, we use interval instead (already set in config)
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        base_work_dir = osp.join('./work_dirs',
                                 osp.splitext(osp.basename(args.config))[0])
        # Add 'test' subdirectory to distinguish from training logs
        cfg.work_dir = osp.join(base_work_dir, 'test')
    else:
        # If work_dir is set in config, append 'test' subdirectory
        cfg.work_dir = osp.join(cfg.work_dir, 'test')

    # default visualization/output dirs if not specified
    if args.show_dir is None:
        args.show_dir = osp.join(cfg.work_dir, 'show')
    if args.out is None:
        args.out = osp.join(cfg.work_dir, 'preds')

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    # MTL의 경우 test_evaluator가 list이므로 처리
    if isinstance(cfg.test_evaluator, list):
        for evaluator in cfg.test_evaluator:
            evaluator['output_dir'] = args.out
            evaluator['keep_results'] = True
    else:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # Test 시 시각화 hook 완전히 비활성화 (크기 불일치 문제 방지)
    # visualization hook이 after_test_iter에서 원본 이미지 크기와 예측 크기 불일치로 에러 발생
    # None으로 설정하면 AssertionError 발생하므로 키 자체를 삭제
    if hasattr(cfg, 'default_hooks') and 'visualization' in cfg.default_hooks:
        del cfg.default_hooks['visualization']

    # Test 시 custom hooks 비활성화 (학습 전용 hooks)
    if hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
