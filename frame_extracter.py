import cv2
import os
import glob

video_folder = r'/home/shinds/my_document/ChamNet/video'
output_dir = r'/home/shinds/my_document/ChamNet/test_frames'
interval = 150  # 30fps 기준 5초 간격

os.makedirs(output_dir, exist_ok=True)

video_paths = glob.glob(os.path.join(video_folder, '*.mp4'))

for video_path in video_paths:
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f'{video_name}_frame_{frame_idx:06d}.jpg')
            success = cv2.imwrite(out_path, frame)
            if success:
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to save: {out_path}")

        frame_idx += 1

    cap.release()

print("모든 영상에서 프레임 추출 완료.")