# 🚀 Jetson Orin High-Performance Inference System

본 프로젝트는 Jetson Orin 환경에서 TensorRT 엔진을 활용하여 **160+ FPS**의 초고속 의미론적 분할(Semantic Segmentation)을 수행하는 시스템입니다.
모든 코드는 `trt_loader.py` 모듈을 기반으로 작동하며, `PyCUDA`를 제거하고 `PyTorch`로 최적화되었습니다.

---

## 📂 파일 구성 및 역할 (File Description)

### 1. 🧠 핵심 코어 (Core Module)
| 파일명 | 역할 |
| :--- | :--- |
| **`test/mtl/trt_loader.py`** | **[핵심]** TensorRT 엔진을 로드하고 GPU 가속 추론을 수행하는 라이브러리 (모든 코드의 심장). |

### 2. 📊 벤치마크 및 테스트 (Benchmark & Test)
| 파일명 | 역할 |
| :--- | :--- |
| **`test/mtl/inference.py`** | 더미 데이터를 사용하여 순수 GPU 연산 속도(FPS)를 측정하는 벤치마크 도구. |
| **`test/zed/camera_test.py`** | ZED 카메라 영상을 받아 추론하고, 결과를 화면에 시각화(Overlay)하는 테스트 코드. |
| **`test/zed/camera_check.py`** | 화면 중앙의 객체가 몇 번 클래스(Class ID)인지 텍스트로 확인하는 디버깅 도구. |

### 3. 📡 응용 및 유틸리티 (Application)
| 파일명 | 역할 |
| :--- | :--- |
| **`test/web_stream.py`** | 별도의 모니터 없이 웹 브라우저(IP:5000)로 로봇의 시야를 원격 관제하는 스트리밍 도구. |
| **`test/zed/drive_recorder.py`** | 실제 주행 데이터를 수집하기 위해 추론된 영상을 AVI 파일로 녹화하는 레코더. |
| **`test/zed/hybrid_stream.py`** | **[추천]** 웹 스트리밍으로 관제하면서 동시에 고화질 녹화를 수행하는 올인원 필드 테스트 도구. |

### 4. 🤖 배포용 (Deployment)
| 파일명 | 역할 |
| :--- | :--- |
| **`test/headless.py`** | 화면 출력(GUI) 없이 오직 카메라 입력과 추론 연산만 반복하는 로봇 탑재용(ROS 2 연동 전) 코드. |

---

## 📐 기술 노트: 해상도 설정 (Resolution Settings)
ZED 2i 카메라는 기본적으로 **Stereo(좌/우 합쳐진)** 영상을 송출하므로, 아래 설정을 준수해야 합니다.

* **입력 해상도 (Input):** `2560 x 720`
    * ZED의 HD720 모드는 좌(1280) + 우(1280)가 합쳐진 Side-by-Side 형식으로 들어옵니다.
    * `cv2.VideoCapture` 설정 시 반드시 `2560`으로 설정해야 정상 작동합니다.
* **전처리 (Preprocessing):** `Left Crop`
    * 코드 내부에서 `frame[:, :1280]`으로 왼쪽 눈 영상만 잘라내어 사용합니다.
* **녹화 해상도 (Recording):** `1280 x 720`
    * 잘라낸 왼쪽 눈 크기에 맞춰 저장되므로, VideoWriter 설정도 1280x720이어야 합니다.

---

## 🚀 실행 방법 (Usage)

**1. 환경 진입 (Docker)**
```bash
chamdog  # 미리 설정된 alias 사용 (chamdog_final 이미지 실행)