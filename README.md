# 🚀 Jetson Orin High-Performance MTL Inference System

본 프로젝트는 Jetson Orin 환경에서 TensorRT 엔진을 활용하여 **Semantic Segmentation + Depth Estimation**을 동시에 수행하는 Multi-Task Learning(MTL) 시스템입니다.
모든 코드는 `trt_loader.py` 모듈을 기반으로 작동하며, `PyCUDA`를 제거하고 `PyTorch`로 최적화되었습니다.

---

## 📂 파일 구성 및 역할 (File Structure)
```
/data/test/
├── trt_loader.py        # [핵심] TensorRT 추론 엔진
├── headless.py          # 로봇 탑재용 (GUI 없음)
├── inference.py         # 벤치마크 도구
├── camera_check.py      # 중앙 클래스/거리 확인
├── camera_test.py       # 카메라 테스트
├── web_stream.py        # 웹 스트리밍
├── drive_recorder.py    # 주행 녹화
└── hybrid_stream.py     # [추천] 스트리밍 + 녹화 + 데이터 저장
```

### 1. 🧠 핵심 코어 (Core Module)
| 파일명 | 역할 |
| :--- | :--- |
| **`trt_loader.py`** | TensorRT 엔진 로드 및 GPU 가속 추론. Segmentation + Depth 동시 출력. |

### 2. 📊 벤치마크 및 테스트 (Benchmark & Test)
| 파일명 | 역할 |
| :--- | :--- |
| **`inference.py`** | 더미 데이터로 순수 GPU 연산 속도(FPS) 측정. |
| **`camera_test.py`** | ZED 카메라 영상 추론 + Seg/Depth 시각화 테스트. |
| **`camera_check.py`** | 화면 중앙의 클래스 ID와 거리(m)를 확인하는 디버깅 도구. |
| **`headless.py`** | GUI 없이 추론만 수행. ROS 2 연동 전 로봇 탑재용. |

### 3. 📡 응용 및 유틸리티 (Application)
| 파일명 | 역할 |
| :--- | :--- |
| **`web_stream.py`** | 웹 브라우저(IP:5000)로 Seg + Depth 원격 관제. |
| **`drive_recorder.py`** | Seg + Depth 합친 영상을 AVI로 녹화. |
| **`hybrid_stream.py`** | **[추천]** 웹 스트리밍 + RGB/Depth NPY 저장. 연구 데이터 수집용. |

---

## 🎯 모델 출력 (Model Output)

| 출력 | Shape | 설명 |
| :--- | :--- | :--- |
| **seg_logits** | `(1, 7, 512, 512)` | 7개 클래스 Segmentation (argmax 후 사용) |
| **depth_map** | `(1, 1, 512, 512)` | Monocular Depth Estimation (미터 단위) |

### 클래스 정의
| ID | 클래스 | 색상 (BGR) |
| :--- | :--- | :--- |
| 0 | Background | 검정 |
| 1 | Chamoe (참외) | 노랑 |
| 2 | Heatpipe (난방관) | 빨강 |
| 3 | Path (통로) | 초록 |
| 4 | Pillar (기둥) | 파랑 |
| 5 | Topdown (상부) | 마젠타 |
| 6 | Unknown | 회색 |

---

## 📐 기술 노트: 해상도 설정 (Resolution Settings)

ZED 2i 카메라는 **Stereo(좌/우 합쳐진)** 영상을 송출합니다.

| 단계 | 해상도 | 설명 |
| :--- | :--- | :--- |
| 입력 (Input) | `2560 x 720` | ZED HD720 Side-by-Side 형식 |
| 전처리 (Crop) | `1280 x 720` | `frame[:, :1280]` 왼쪽 눈만 사용 |
| 모델 입력 | `512 x 512` | 내부 리사이즈 |
| 녹화 (Seg+Depth) | `2560 x 720` | 좌: Seg, 우: Depth |

---

## 🚀 실행 방법 (Usage)

### 1. 환경 진입 (Docker)
```bash
cd ~/model/jetson-containers
./run.sh $(./autotag pytorch) bash
```

### 2. 테스트 실행
```bash
# 순수 추론 속도 테스트 (GUI 없음)
python /data/test/headless.py

# 웹 스트리밍 (브라우저에서 http://<ORIN_IP>:5000 접속)
python /data/test/web_stream.py

# 연구용 데이터 수집 (RGB + Depth NPY 저장)
python /data/test/hybrid_stream.py
```

### 3. 저장 데이터 구조 (hybrid_stream.py)
```
/data/recordings/20260114_160000/
├── rgb/
│   ├── 000000.png
│   └── ...
├── pred_depth/
│   ├── 000000.npy      # float32, 미터 단위
│   └── ...
├── pred_seg/
│   ├── 000000.npy      # uint8, 클래스 인덱스
│   └── ...
└── meta.npy            # 총 프레임 수, 타임스탬프
```

---

## ⚡ 성능 (Performance on AGX Orin 64GB)

| 항목 | 수치 |
| :--- | :--- |
| 순수 GPU 추론 | **~98 FPS** |
| 카메라 포함 루프 | **~30 FPS** |
| 웹 스트리밍 | **~12 FPS** |
| 추론 지연시간 | **~10 ms** |

---

## 🔧 트러블슈팅 (Troubleshooting)

### OOM (Out of Memory) 에러
```bash
# 이전 프로세스가 GPU 메모리 점유 중일 수 있음
pkill -9 python
python /data/test/headless.py
```

### 카메라 연결 실패
```bash
# USB 뺐다 다시 꽂기 또는:
sudo udevadm trigger
```

### Docker 안에서 카메라 안 잡힘
```bash
# 컨테이너 나갔다 다시 실행
exit
./run.sh $(./autotag pytorch) bash
```

---

## 📝 참고사항

- TensorRT 엔진은 **빌드 환경과 실행 환경이 동일**해야 합니다.
- Docker 안에서 빌드한 `.engine` 파일은 Docker 안에서만 실행 가능합니다.
- ZED SDK는 현재 Docker 이미지에 미포함. SVO 녹화가 필요하면 Host에서 별도 실행하세요.