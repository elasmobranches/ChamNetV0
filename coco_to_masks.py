import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
'''
이 스크립트는 COCO 형식의 JSON 어노테이션을 세그멘테이션 학습용 단일 채널 마스크와 시각화용 컬러 마스크로 변환하는 도구입니다. 
특히, 여러 객체가 겹쳐 있을 때 클래스별 우선순위에 따라 라벨을 할당하는 기능을 지원합니다.
'''
# 시각화를 위한 컬러 맵 지정
color_map = {
    0: (0, 0, 0),        # background - black
    1: (255, 255, 0),    # chamoe - yellow
    2: (255, 0, 0),      # heatpipe - red
    3: (0, 255, 0),      # path - green
    4: (0, 0, 255),      # pillar - blue
    5: (255, 0, 255),    # topdownfarm - magenta
    6: (128, 128, 128),  # unknown - gray
}
def normalize_name(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name.strip().lower())


def build_category_to_label_map(categories: List[dict]) -> Dict[int, int]:
    normalized_name_to_label = {
        "background": 0,
        "chamoe": 1,
        "heatpipe": 2,
        "path": 3,
        "pillar": 4,
        "topdownfarm": 5,
        "unknown": 6,
    }

    cat_id_to_label: Dict[int, int] = {}
    for cat in categories:
        cat_id = int(cat["id"])  # COCO category id
        name_norm = normalize_name(cat.get("name", ""))
        if name_norm in normalized_name_to_label:
            cat_id_to_label[cat_id] = normalized_name_to_label[name_norm]
        else:
            # 알 수 없는 클래스명은 0(background)으로 매핑
            cat_id_to_label[cat_id] = 0
    return cat_id_to_label


def polygon_to_mask(height: int, width: int, polygons: List[List[float]]) -> np.ndarray:
    from PIL import Image, ImageDraw

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # COCO 폴리곤은 [x1,y1,x2,y2,...] 리스트를 1개 또는 다수 포함
    for poly in polygons:
        if len(poly) < 6:
            continue
        xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_img, dtype=bool)


def try_decode_rle(height: int, width: int, segmentation: dict) -> np.ndarray:
    try:
        from pycocotools import mask as maskUtils  # type: ignore
        rle = segmentation
        if "counts" in rle and isinstance(rle["counts"], list):
            # uncompressed RLE -> COCO RLE로 변환
            rle = maskUtils.frPyObjects([rle], height, width)[0]
        m = maskUtils.decode(rle)
        return m.astype(bool)
    except Exception:
        raise NotImplementedError("RLE segmentation requires pycocotools. Please install it or convert to polygons.")


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if seg is None:
        return np.zeros((height, width), dtype=bool)

    if isinstance(seg, list):
        return polygon_to_mask(height, width, seg)

    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        return try_decode_rle(height, width, seg)

    return np.zeros((height, width), dtype=bool)


def parse_priority(priority_str: str) -> List[int]:
    """
    우선순위 문자열을 파싱해서 라벨 번호 리스트로 반환.
    - 기본 우선순위: [3, 5, 1, 4, 2, 6]
    - 입력은 "4,3,5,1,6,2" 같은 숫자 리스트만 허용 (숫자 아닌 항목은 무시)
    - 중복은 제거하고 입력 순서를 유지
    - 잘못된 입력(비어있거나 파싱 실패)이면 기본값 반환
    """
    default_priority = [3, 5, 2, 1, 4, 6, 7]
# ''''"background": 0,
# ''''"chamoe": 1,
# ''''"heatpipe": 2,
# ''''"path": 3,
# ''''"pillar": 4,
# ''''"topdownfarm": 5,
#     "unknown": 6
# duct : 7

    if not priority_str:
        return default_priority

    parts = [p.strip() for p in priority_str.split(",") if p.strip() != ""]
    out: List[int] = []
    seen = set()
    for p in parts:
        try:
            n = int(p)
        except ValueError:
            # 숫자가 아니면 무시
            print(f"[경고] priority 항목이 정수가 아님, 무시: {p}")
            continue
        if n in seen:
            # 중복 무시
            continue
        out.append(n)
        seen.add(n)

    return out or default_priority


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def labels_to_color_image(
    label_mask: np.ndarray,
    label_to_color: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """라벨 마스크(HxW, uint8)를 RGB 컬러 이미지(HxWx3, uint8)로 변환."""
    height, width = label_mask.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value, rgb in label_to_color.items():
        color_img[label_mask == label_value] = rgb
    return color_img


def convert_coco_to_gray_masks(
    coco_json_path: str,
    out_dir: str,
    priority_str: str,
    color_out_dir: Optional[str] = None,
) -> None:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_id_to_label = build_category_to_label_map(categories)
    label_priority = parse_priority(priority_str)

    # 이미지 메타
    img_by_id: Dict[int, Tuple[int, int, str]] = {}
    for img in images:
        img_id = int(img["id"])
        width = int(img["width"])
        height = int(img["height"])
        file_name = img["file_name"]
        img_by_id[img_id] = (width, height, file_name)

    # 이미지별 주석 그룹화
    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_img[int(ann["image_id"])].append(ann)

    ensure_dir(out_dir)
    # 컬러 출력 디렉토리 기본값 및 생성
    if color_out_dir is None:
        color_out_dir = os.path.join(os.path.dirname(coco_json_path), "masks_color")
    ensure_dir(color_out_dir)

    # 라벨별로 해당 coco category ids 역인덱스
    label_to_cat_ids: Dict[int, List[int]] = defaultdict(list)
    for cat_id, label in cat_id_to_label.items():
        label_to_cat_ids[label].append(cat_id)

    for img_id, (width, height, file_name) in img_by_id.items():
        final_mask = np.zeros((height, width), dtype=np.uint8)
        img_anns = anns_by_img.get(img_id, [])

        # 우선순위대로 합성: 먼저 그린 것이 나중 레이블에 의해 덮임
        for label in label_priority:
            cat_ids = set(label_to_cat_ids.get(label, []))
            if not cat_ids:
                continue
            for ann in img_anns:
                if int(ann.get("category_id", -1)) not in cat_ids:
                    continue
                m = ann_to_mask(ann, height, width)
                if m is None:
                    continue
                final_mask[m] = np.uint8(label)

        # 파일명 규칙: <베이스>_mask.png
        base, _ = os.path.splitext(os.path.basename(file_name))
        out_path = os.path.join(out_dir, f"{base}_mask.png")
        color_out_path = os.path.join(color_out_dir, f"{base}_mask.png")

        try:
            import cv2  # type: ignore

            cv2.imwrite(out_path, final_mask)
            # 컬러 마스크 저장 (RGB -> BGR 변환 후 저장)
            color_img = labels_to_color_image(final_mask, color_map)
            color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(color_out_path, color_img_bgr)
        except Exception:
            from PIL import Image

            Image.fromarray(final_mask, mode="L").save(out_path)
            color_img = labels_to_color_image(final_mask, color_map)
            Image.fromarray(color_img, mode="RGB").save(color_out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to single-channel gray masks with class-priority overlap handling."
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        required=True,
        help="Path to _annotations.coco.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for masks (default: <coco_json_dir>/masks)",
    )
    parser.add_argument(
        "--color-out-dir",
        type=str,
        default=None,
        help="Output directory for color masks (default: <coco_json_dir>/masks_color)",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="4,3,5,1,6,2",
        help="Comma-separated label priority (numbers only). Example: '4,3,5,1,6,2'. Later labels overwrite earlier ones.",
    )

    args = parser.parse_args()

    coco_json_path = os.path.abspath(args.coco_json)
    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.join(os.path.dirname(coco_json_path), "masks_gray")
    )

    convert_coco_to_gray_masks(
        coco_json_path=coco_json_path,
        out_dir=out_dir,
        priority_str=args.priority,
        color_out_dir=args.color_out_dir,
    )


if __name__ == "__main__":
    main()
