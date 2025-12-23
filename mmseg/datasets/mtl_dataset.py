"""
MTL Dataset for MMSegmentation

Depth 데이터를 포함하는 Multi-Task Learning 데이터셋을 구현합니다.
"""

import os
import warnings
from typing import Dict, List

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class MTLChamDataset(BaseSegDataset):
    """Multi-Task Learning Dataset for ChamData

    Segmentation과 Depth 데이터를 모두 로드하는 데이터셋입니다.

    Args:
        data_root (str): 데이터 루트 디렉토리
        data_prefix (dict): 데이터 경로 prefix
            - img_path: 이미지 경로
            - seg_map_path: Segmentation 맵 경로
            - depth_map_path: Depth 맵 경로
        pipeline (List[dict]): 데이터 변환 파이프라인
        img_suffix (str): 이미지 파일 확장자
        seg_map_suffix (str): Segmentation 맵 파일 확장자
        depth_map_suffix (str): Depth 맵 파일 확장자
    """

    METAINFO = dict(
        classes=('background', 'chamoe', 'heatpipe', 'path', 'pillar', 
                 'topdownfarm', 'unknown'),
        palette=[[0, 0, 0],          # background - black
                 [255, 255, 0],      # chamoe - yellow
                 [255, 0, 0],        # heatpipe - red
                 [0, 255, 0],        # path - green
                 [0, 0, 255],        # pillar - blue
                 [255, 0, 255],      # topdownfarm - magenta
                 [128, 128, 128]]    # unknown - gray
    )

    def __init__(self,
                 data_root: str = '',
                 data_prefix: dict = dict(
                     img_path='',
                     seg_map_path='',
                     depth_map_path=''
                 ),
                 pipeline: List[Dict] = [],
                 img_suffix: str = '.jpg',
                 seg_map_suffix: str = '.png',
                 depth_map_suffix: str = '.png',
                 **kwargs) -> None:

        self.depth_map_suffix = depth_map_suffix
        self.data_prefix = data_prefix

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )

    def load_data_list(self) -> List[Dict]:
        """데이터 리스트 로드

        이미지, segmentation, depth 파일 경로를 매칭하여 리스트를 생성합니다.

        Returns:
            List[Dict]: 데이터 정보 리스트
        """
        data_list = []

        # data_prefix는 이미 data_root와 결합되어 있음 (_join_prefix()에 의해)
        img_dir = self.data_prefix.get('img_path', None)
        seg_dir = self.data_prefix.get('seg_map_path', None)
        depth_dir = self.data_prefix.get('depth_map_path', None)

        # 이미지 파일 리스트
        _suffix_len = len(self.img_suffix)
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.endswith(self.img_suffix):
                continue

            # 파일 이름 (확장자 제외)
            file_name = img_file[:-_suffix_len]

            # 경로 생성
            img_path = os.path.join(img_dir, img_file)
            seg_path = os.path.join(seg_dir, file_name + self.seg_map_suffix)
            depth_path = os.path.join(depth_dir, file_name + self.depth_map_suffix)

            # 파일 존재 확인
            if not os.path.exists(img_path):
                warnings.warn(f"Image file not found: {img_path}")
                continue

            if not os.path.exists(seg_path):
                warnings.warn(f"Segmentation file not found: {seg_path}")
                continue

            # Depth 파일은 선택적 (없으면 경고만)
            if not os.path.exists(depth_path):
                warnings.warn(f"Depth file not found: {depth_path}, will use zeros")
                depth_path = None

            # 데이터 정보 딕셔너리
            data_info = dict(
                img_path=img_path,
                seg_map_path=seg_path,
                depth_map_path=depth_path,
                reduce_zero_label=False,  # 0번 클래스도 유효한 클래스
                seg_fields=[]
            )

            data_list.append(data_info)

        return data_list
