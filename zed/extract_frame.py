import pyzed.sl as sl
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
def extract_svo_frames(svo_path, output_dir, sample_rate=10):
    """
    SVO에서 RGB + Depth 추출 (depth mode 보존)
    """
    os.makedirs(f"{output_dir}/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/depth", exist_ok=True)
    os.makedirs(f"{output_dir}/depth_visual", exist_ok=True)
    os.makedirs(f"{output_dir}/confidence", exist_ok=True)
    os.makedirs(f"{output_dir}/confidence_visual", exist_ok=True)
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    
    # ===== 중요: Depth mode를 녹화 시와 동일하게 설정 =====
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # 또는 NEURAL
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = 200
    init_params.depth_maximum_distance = 20000
    init_params.depth_stabilization = 1
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"SVO 열기 실패: {err}")
        return
    
    # 실제 사용된 depth mode 확인
    actual_depth_mode = zed.get_init_parameters().depth_mode
    print(f"Depth Mode: {actual_depth_mode}")
    
    nb_frames = zed.get_svo_number_of_frames()
    print(f"총 프레임: {nb_frames}")
    print(f"샘플링: {sample_rate}프레임당 1개")
    print(f"추출 예상: {nb_frames // sample_rate}개\n")
    
    # Runtime parameters - 녹화 시와 동일하게 안 같으면 동일하게 안 뽑힙니다..
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 0  # 모든 depth 데이터 가져오기
    runtime.texture_confidence_threshold = 0
    runtime.enable_depth = True  # 명시적으로 depth 활성화
    runtime.enable_fill_mode = True  # 원본 데이터 유지
    
    image_left = sl.Mat()
    depth_map = sl.Mat()
    confidence_map = sl.Mat()
    
    frame_count = 0
    saved_count = 0
    depth_stats = []
    
    pbar = tqdm(total=nb_frames, desc="프레임 추출")
    
    while True:
        err = zed.grab(runtime)
        
        if err == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            pbar.update(1)
            
            # 샘플링
            if frame_count % sample_rate != 0:
                continue
            
            # RGB 추출
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            rgb = image_left.get_data()
            
            # Depth 추출
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU)
            depth = depth_map.get_data()
            
            # Confidence 추출
            zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE, sl.MEM.CPU)
            confidence = confidence_map.get_data()
            
            # Depth 유효성 검증
            valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0)
            valid_count = valid_mask.sum()
            total_count = depth.size
            
            if saved_count == 0 or saved_count % 10 == 0:
                print(f"\n프레임 {frame_count}: 유효 depth 픽셀 {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                if valid_count > 0:
                    valid_depth = depth[valid_mask]
                    print(f"  Min: {valid_depth.min()/1000:.2f}m, Max: {valid_depth.max()/1000:.2f}m, Mean: {valid_depth.mean()/1000:.2f}m")
                
                # 첫 프레임에서 Confidence 범위 확인
                if saved_count == 0:
                    print(f"  Confidence 범위: {confidence.min():.1f} ~ {confidence.max():.1f}")
            
            # BGR 변환
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
            
            # 저장
            frame_id = f"{saved_count:06d}"
            cv2.imwrite(f"{output_dir}/rgb/{frame_id}.png", rgb_bgr)
            
            # Depth 저장 (uint16으로 저장 - mm 단위라 소수점 필요 없음)
            # 0~65535mm (65m) 범위 커버, 1mm 정밀도
            depth_valid = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
            depth_uint16 = np.clip(depth_valid, 0, 65535).astype(np.uint16)
            np.save(f"{output_dir}/depth/{frame_id}.npy", depth_uint16)
            
            # Confidence 저장 (0-255 범위, uint8로 충분)
            confidence_uint8 = np.clip(confidence, 0, 255).astype(np.uint8)
            np.save(f"{output_dir}/confidence/{frame_id}.npy", confidence_uint8)
            
            # Confidence 시각화
            conf_visual_color = cv2.applyColorMap(confidence_uint8, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f"{output_dir}/confidence_visual/{frame_id}.png", conf_visual_color)
            
            # Depth 시각화 (mm를 m로 변환)
            depth_m = depth / 1000.0  # mm to m
            depth_visual = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 유효한 depth 값만으로 시각화 범위 설정
            if valid_count > 0:
                valid_depth_m = depth_m[valid_mask]
                min_d = np.percentile(valid_depth_m, 5)
                max_d = np.percentile(valid_depth_m, 95)
                depth_visual = np.clip(depth_visual, min_d, max_d)
                depth_visual = ((depth_visual - min_d) / (max_d - min_d) * 255).astype(np.uint8)
            else:
                depth_visual = np.zeros_like(depth_visual, dtype=np.uint8)
            
            depth_visual_color = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/depth_visual/{frame_id}.png", depth_visual_color)
            
            # 통계 저장
            if valid_count > 0:
                valid_conf = confidence[valid_mask]
                depth_stats.append({
                    'frame': frame_count,
                    'valid_ratio': valid_count / total_count,
                    'min_depth': valid_depth.min() / 1000.0,
                    'max_depth': valid_depth.max() / 1000.0,
                    'mean_depth': valid_depth.mean() / 1000.0,
                    'mean_confidence': valid_conf.mean()
                })
            
            saved_count += 1
            
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("\nSVO 파일 끝 도달")
            break
        else:
            print(f"\nGrab 에러: {err}")
            break
    
    pbar.close()
    zed.close()
    
    print(f"\n추출 완료!")
    print(f"- 총 프레임: {frame_count}")
    print(f"- 저장: {saved_count}개")
    print(f"- 위치: {output_dir}/")
    
    # Depth 통계 요약
    if depth_stats:
        avg_valid = np.mean([s['valid_ratio'] for s in depth_stats])
        avg_confidence = np.mean([s['mean_confidence'] for s in depth_stats])
        print(f"\nDepth 통계:")
        print(f"- 평균 유효 픽셀 비율: {avg_valid*100:.1f}%")
        print(f"- 평균 depth 범위: {np.mean([s['min_depth'] for s in depth_stats]):.2f}m ~ {np.mean([s['max_depth'] for s in depth_stats]):.2f}m")
        print(f"- 평균 confidence: {avg_confidence:.1f}")
    
    # 메타데이터 저장
    metadata = {
        "total_frames": frame_count,
        "saved_frames": saved_count,
        "sample_rate": sample_rate,
        "depth_stats": depth_stats
    }
    np.save(f"{output_dir}/metadata.npy", metadata)

parser = argparse.ArgumentParser()
parser.add_argument("--svo_path", type=str,required = True)
parser.add_argument("--output_dir", type=str, required=False, default="./extracted_frames/new")
parser.add_argument("--sample_rate", type=int, default=10)
args = parser.parse_args()
    
extract_svo_frames(args.svo_path, args.output_dir, args.sample_rate)