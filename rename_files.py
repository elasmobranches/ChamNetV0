#!/usr/bin/env python3
"""
파일명 단순화 스크립트

복잡한 파일명을 간단하게 변경:
- images: 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc.jpg 
  → 20250526_rfv4_frame_000298_00m_09s.jpg

- masks: 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_mask.png
  → 20250526_rfv4_frame_000298_00m_09s_mask.png

- depth: 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_depth.png
  → 20250526_rfv4_frame_000298_00m_09s_depth.png
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple


def simplify_filename(filename: str, file_type: str) -> str:
    """
    파일명을 단순화합니다.
    
    Args:
        filename: 원본 파일명
        file_type: 'image', 'mask', 'depth' 중 하나
        
    Returns:
        단순화된 파일명
    """
    # 확장자 분리
    name, ext = os.path.splitext(filename)
    
    if file_type == 'image':
        # images: jpg 이후 부분 제거 (증강 파일 포함)
        # 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc
        # → 20250526_rfv4_frame_000298_00m_09s
        # 250731output_video_250404_chamoe1355_frame_000006_jpg.rf.57adef4cf58462ffef88129ba7463a71_aug_001
        # → 250731output_video_250404_chamoe1355_frame_000006
        match = re.match(r'^(.+)_jpg\.rf\..+(_aug_\d+)?$', name)
        if match:
            return f"{match.group(1)}{ext}"
        return filename
    
    elif file_type == 'mask':
        # masks: jpg부터 mask 앞까지 제거 (증강 파일 포함)
        # 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_mask
        # → 20250526_rfv4_frame_000298_00m_09s_mask
        # 250731output_video_250404_chamoe1355_frame_000006_jpg.rf.57adef4cf58462ffef88129ba7463a71_aug_001_mask
        # → 250731output_video_250404_chamoe1355_frame_000006_mask
        match = re.match(r'^(.+)_jpg\.rf\..+(_aug_\d+)?_mask$', name)
        if match:
            return f"{match.group(1)}_mask.png"  # 확장자를 .png로 변경
        
        # 이미 단순화된 파일들 처리 (mask_001.jpg → mask.png)
        match = re.match(r'^(.+)_mask_\d+$', name)
        if match:
            return f"{match.group(1)}_mask.png"  # 확장자를 .png로 변경
        
        return filename

    elif file_type == 'mask_gray':
        # mask_gray: jpg부터 mask_gray 앞까지 제거 (증강 파일 포함)
        # 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_mask_gray
        # → 20250526_rfv4_frame_000298_00m_09s_mask_gray
        # 250731output_video_250404_chamoe1355_frame_000006_jpg.rf.57adef4cf58462ffef88129ba7463a71_aug_001_mask_gray
        # → 250731output_video_250404_chamoe1355_frame_000006_mask_gray
        match = re.match(r'^(.+)_jpg\.rf\..+(_aug_\d+)?_mask_gray$', name)
        if match:
            return f"{match.group(1)}_mask_gray.png"  # 확장자를 .png로 변경
        return filename
    elif file_type == 'mask_color':
        # mask_color: jpg부터 mask_color 앞까지 제거 (증강 파일 포함)
        # 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_mask_color
        # → 20250526_rfv4_frame_000298_00m_09s_mask_color
        # 250731output_video_250404_chamoe1355_frame_000006_jpg.rf.57adef4cf58462ffef88129ba7463a71_aug_001_mask_color
        # → 250731output_video_250404_chamoe1355_frame_000006_mask_color
        match = re.match(r'^(.+)_jpg\.rf\..+(_aug_\d+)?_mask_color$', name)
        if match:
            return f"{match.group(1)}_mask_color.png"  # 확장자를 .png로 변경
        return filename
    elif file_type == 'depth':
        # depth: jpg부터 depth 앞까지 제거 (증강 파일 포함)
        # 20250526_rfv4_frame_000298_00m_09s_jpg.rf.76b2eb06aa974a44f32ae875a3e30ffc_depth
        # → 20250526_rfv4_frame_000298_00m_09s_depth
        # 250731output_video_250404_chamoe1355_frame_000006_jpg.rf.57adef4cf58462ffef88129ba7463a71_aug_000_depth
        # → 250731output_video_250404_chamoe1355_frame_000006_depth
        match = re.match(r'^(.+)_jpg\.rf\..+(_aug_\d+)?_depth$', name)
        if match:
            return f"{match.group(1)}_depth{ext}"
        return filename


    
    return filename


def rename_files_in_directory(directory: Path, file_type: str, dry_run: bool = False) -> List[Tuple[str, str]]:
    """
    디렉토리 내의 파일들을 이름을 변경합니다.
    
    Args:
        directory: 대상 디렉토리
        file_type: 'image', 'mask', 'depth' 중 하나
        dry_run: True면 실제 변경하지 않고 미리보기만
        
    Returns:
        (원본파일명, 새파일명) 튜플 리스트
    """
    if not directory.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    renamed_files = []
    used_names = set()  # 이미 사용된 파일명 추적
    
    # 파일 확장자 결정
    if file_type == 'image':
        extensions = ['.jpg', '.jpeg', '.png']
    elif file_type == 'depth':
        extensions = ['.jpg', '.jpeg', '.png']  # depth 파일도 jpg 확장자 사용
    elif file_type == 'mask':
        extensions = ['.jpg', '.jpeg', '.png']  # mask 파일도 jpg 확장자 사용
    elif file_type == 'mask_gray':
        extensions = ['.jpg', '.jpeg', '.png']  # mask 파일도 jpg 확장자 사용
    elif file_type == 'mask_color':
        extensions = ['.jpg', '.jpeg', '.png']  # mask 파일도 jpg 확장자 사용
    else:
        extensions = ['.png']
    
    for ext in extensions:
        for file_path in directory.glob(f'*{ext}'):
            old_name = file_path.name
            new_name = simplify_filename(old_name, file_type)
            
            if old_name != new_name:
                # 파일명 충돌 해결: 증강 파일에 번호 추가
                original_new_name = new_name
                counter = 1
                while new_name in used_names:
                    name_part, ext_part = os.path.splitext(original_new_name)
                    new_name = f"{name_part}_{counter:03d}{ext_part}"
                    counter += 1
                
                used_names.add(new_name)
                renamed_files.append((old_name, new_name))
                
                if not dry_run:
                    old_path = file_path
                    new_path = file_path.parent / new_name
                    
                    try:
                        old_path.rename(new_path)
                        if original_new_name != new_name:
                            print(f"✅ {old_name} → {new_name} (충돌 해결)")
                        else:
                            print(f"✅ {old_name} → {new_name}")
                    except Exception as e:
                        print(f"❌ {old_name} 변경 실패: {e}")
                else:
                    if original_new_name != new_name:
                        print(f"🔍 {old_name} → {new_name} (충돌 해결)")
                    else:
                        print(f"🔍 {old_name} → {new_name}")
    
    return renamed_files


def main():
    parser = argparse.ArgumentParser(
        description="복잡한 파일명을 단순화합니다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python rename_files.py --dataset_root /path/to/dataset --dry_run
  python rename_files.py --dataset_root /path/to/dataset --types image mask depth
        """
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="데이터셋 루트 디렉토리 경로"
    )
    
    parser.add_argument(
        "--types",
        nargs='+',
        choices=['image', 'mask', 'depth', 'mask_gray', 'mask_color'],
        default=['image', 'mask', 'depth', 'mask_gray', 'mask_color'],
        help="변경할 파일 타입들 (기본값: 모든 타입)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실제 변경하지 않고 미리보기만 실행"
    )
    
    parser.add_argument(
        "--splits",
        nargs='+',
        choices=['train', 'val', 'test'],
        default=['train', 'val', 'test'],
        help="처리할 split들 (기본값: 모든 split)"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ 데이터셋 루트가 존재하지 않습니다: {dataset_root}")
        return
    
    print(f"🚀 파일명 단순화 시작...")
    print(f"📁 데이터셋 루트: {dataset_root}")
    print(f"📝 처리할 타입: {', '.join(args.types)}")
    print(f"📂 처리할 split: {', '.join(args.splits)}")
    if args.dry_run:
        print("🔍 미리보기 모드 (실제 변경하지 않음)")
    print()
    
    total_renamed = 0
    
    for split in args.splits:
        print(f"\n📂 {split} split 처리 중...")
        
        for file_type in args.types:
            if file_type == 'image':
                target_dir = dataset_root / split / 'images'
            elif file_type == 'mask':
                target_dir = dataset_root / split / 'masks'
            elif file_type == 'depth':
                target_dir = dataset_root / split / 'depth'
            elif file_type == 'mask_gray':
                target_dir = dataset_root / split / 'masks_gray'
            elif file_type == 'mask_color':
                target_dir = dataset_root / split / 'masks_color'
            else:
                continue
            
            print(f"  📁 {file_type} 파일들 처리 중: {target_dir}")
            renamed_files = rename_files_in_directory(target_dir, file_type, args.dry_run)
            total_renamed += len(renamed_files)
            
            if renamed_files:
                print(f"    📊 {len(renamed_files)}개 파일 {'변경 예정' if args.dry_run else '변경 완료'}")
            else:
                print(f"    ℹ️  변경할 파일이 없습니다")
    
    print(f"\n🎉 완료!")
    print(f"📊 총 {total_renamed}개 파일 {'변경 예정' if args.dry_run else '변경 완료'}")


if __name__ == "__main__":
    main()
