# ZED ArUco Depth Estimation v2.0

ZED 2i 카메라와 ArUco 마커를 이용한 거리 측정 시스템 (Flask 웹 스트리밍 버전)

## 주요 특징

- **Flask 웹 스트리밍**: WiFi 없는 환경에서도 웹 브라우저로 접근 가능
- **모듈화된 구조**: 카메라, 마커 감지, depth 분석이 독립적인 모듈로 분리
- **실시간 시각화**: 브라우저에서 실시간으로 카메라 영상과 측정 결과 확인
- **키보드 제어**: 웹 UI에서 키보드 단축키로 빠른 조작
- **자동 CSV 저장**: 측정 기록을 자동으로 CSV 파일로 저장
- **강력한 로깅**: 체계적인 로그 시스템으로 디버깅 용이

## 📋 필요 사항

### 하드웨어
- ZED 2i 카메라
- ArUco 마커 (DICT_4X4_50, ID: 30, 크기: 100mm)
- 레이저 거리 측정기 (실제 거리 측정용)

### 소프트웨어
- Python 3.7+
- ZED SDK 4.0+ ([다운로드](https://www.stereolabs.com/developers/release/))
- OpenCV 4.5+
- Flask 2.0+

## 🚀 설치

### 1. ZED SDK 설치
```bash
# ZED SDK 다운로드 및 설치 (공식 웹사이트에서)
# https://www.stereolabs.com/developers/release/
```

### 2. Python 패키지 설치
```bash
cd /home/cv_test/Desktop/depth_estimation
pip install -r requirements.txt
```

### 3. 설정 파일 확인
[config/config.yaml](config/config.yaml)에서 설정을 확인하고 필요시 수정합니다.

```yaml
# 주요 설정 항목
marker_id: 30              # ArUco 마커 ID
web_server:
  host: 0.0.0.0           # 모든 네트워크에서 접속 허용
  port: 5000              # 웹 서버 포트
```

## 📖 사용법

### 1. 애플리케이션 실행
```bash
# 기본 실행
python3 main.py

# 커스텀 설정 파일 사용
python3 main.py -c my_config.yaml

# 포트 변경
python3 main.py -p 8080

# 호스트 변경
python3 main.py -H 192.168.1.100
```

### 2. 웹 브라우저 접속
```
http://[서버IP]:5000
```

예시:
- 로컬: `http://localhost:5000`
- 원격: `http://192.168.1.100:5000`

### 3. 측정 절차

1. **마커 배치**: ArUco 마커를 평평한 표면에 부착
2. **거리 측정**: 레이저 측정기로 ZED 카메라와 마커 간 거리 측정
3. **거리 입력**: 웹 UI에서 측정한 거리 입력 (또는 `D` 키)
4. **프레임 저장**: 마커가 잘 감지되면 `S` 키 또는 "프레임 저장" 버튼 클릭
5. **반복**: 다양한 거리에서 측정 반복
6. **종료**: `Q` 키 또는 "종료" 버튼 클릭

### 4. 키보드 단축키

| 키 | 기능 |
|---|---|
| `D` | 거리 입력란 포커스 |
| `S` | 현재 프레임 저장 |
| `M` | 측정 모드 변경 (마커 영역 ↔ 윈도우) |
| `Q` | 종료 |

## 📁 프로젝트 구조

```
depth_estimation/
├── app/
│   ├── camera/          # ZED 카메라 모듈
│   │   └── zed_camera.py
│   ├── marker/          # ArUco 마커 감지 모듈
│   │   └── aruco_detector.py
│   ├── measurement/     # Depth 분석 모듈
│   │   └── depth_analyzer.py
│   ├── web/            # Flask 웹 서버
│   │   └── stream_server.py
│   └── utils/          # 유틸리티
│       ├── config.py
│       └── logger.py
├── config/             # 설정 파일
│   └── config.yaml
├── static/             # 웹 정적 파일
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── stream.js
├── templates/          # HTML 템플릿
│   └── index.html
├── data/              # 출력 데이터
│   └── results/       # CSV 및 이미지
├── logs/              # 로그 파일
├── main.py            # 메인 애플리케이션
├── requirements.txt   # Python 의존성
└── README.md
```

## 📊 출력 파일

### CSV 파일 (data/results/)
측정 기록이 CSV 파일로 저장됩니다:
```csv
timestamp,actual_m,zed_median_mm,zed_mean_mm,zed_std_mm,error_mm,error_pct,marker_angle,valid_ratio
2024-01-31 19:30:45.123,1.500,1485.3,1486.1,12.5,14.7,0.98,2.3,0.95
```

### 이미지 (data/results/images/)
저장된 프레임 이미지:
```
1.500m_193045.jpg
2.000m_193112.jpg
...
```

### 로그 (logs/)
애플리케이션 로그:
```
depth_estimation_20240131.log
```

## 🔧 설정 가이드

### Depth 측정 모드

#### 1. 마커 영역 모드 (권장)
```yaml
depth_measurement:
  use_marker_region: true
```
- 마커 영역 전체의 depth 값을 사용
- IQR 기반 아웃라이어 제거
- 더 정확하고 안정적

#### 2. 윈도우 모드
```yaml
depth_measurement:
  use_marker_region: false
  window_size: 11
```
- 마커 중심점 주변 윈도우만 사용
- 빠르지만 덜 정확

### ZED 카메라 설정

```yaml
zed_settings:
  depth_mode: NEURAL_PLUS      # 최고 정확도
  resolution: HD1080           # 높은 해상도
  fps: 30                      # 부드러운 스트리밍
  depth_stabilization: 1       # Depth 안정화
```

### 웹 서버 설정

```yaml
web_server:
  host: 0.0.0.0       # 모든 IP에서 접속 허용
  port: 5000          # 포트 번호
  stream_quality: 85  # JPEG 품질 (높을수록 선명, 느림)
  stream_fps: 15      # 스트리밍 FPS (낮을수록 부드러움)
```

## 🔍 노출 디버깅 도구 (debug_exposure.py)

마커 인식률을 높이기 위한 **카메라 노출/게인 최적화 도구**입니다.

### 왜 필요한가?

ZED 카메라의 Auto Exposure는 전체 화면 밝기를 기준으로 동작하여, 특정 조명 환경에서 ArUco 마커가 제대로 감지되지 않을 수 있습니다.
이 도구를 사용하면:
- 🔧 실시간으로 노출/게인/밝기/대비/선명도 조정
- 📊 마커 검출 통계를 실시간으로 확인
- 💾 최적 설정값을 저장하여 main.py에 적용

### 사용 방법

#### 1단계: 디버깅 도구 실행

```bash
# 기본 실행 (포트 5001)
python3 debug_exposure.py

# 커스텀 포트
python3 debug_exposure.py -p 8080

# 커스텀 설정 파일
python3 debug_exposure.py -c config/config.yaml
```

#### 2단계: 웹 브라우저 접속

```
http://[서버IP]:5001
```

#### 3단계: 최적 설정 찾기

1. **Auto 모드 테스트**
   - 기본적으로 Auto Exposure 모드로 시작
   - 마커 검출률 확인

2. **Manual 모드로 전환**
   - "🔄 Manual 모드로 전환" 버튼 클릭
   - 노출(Exposure)과 게인(Gain) 슬라이더 활성화

3. **값 조정**
   - **노출(Exposure)**: 0-100 (높을수록 밝음, 보통 40-70 권장)
   - **게인(Gain)**: 0-100 (높을수록 밝지만 노이즈 증가, 보통 30-60 권장)
   - **밝기(Brightness)**: 0-8 (이미지 후처리 밝기)
   - **대비(Contrast)**: 0-8 (명암 대비)
   - **선명도(Sharpness)**: 0-8 (엣지 강조)

4. **목표 달성**
   - 마커 검출률 **90% 이상** 달성 시까지 조정

5. **설정 저장**
   - "💾 설정 저장" 버튼 클릭
   - 터미널에 권장 설정값 출력됨

#### 4단계: main.py에 최적 설정 적용

터미널 로그에 출력된 값을 확인:

```
============================================================
📝 현재 설정 저장(예시)
============================================================
모드: MANUAL
노출: 65
게인: 45
밝기: 5
대비: 4
선명도: 6
평균 밝기: 128.5
마커 검출률: 95.2%
============================================================
```

[app/camera/zed_camera.py](app/camera/zed_camera.py)의 `open()` 메서드 **line 136**에 추가:

```python
self._is_opened = True

# ========== 최적 카메라 설정 적용 (2024-XX-XX 조명 환경 기준) ==========
# debug_exposure.py로 찾은 최적값 - 마커 검출률 95% 달성
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)        # Auto Exposure 끄기
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 65)     # 노출값
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 45)         # 게인값
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)    # 밝기
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)      # 대비
self.camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)     # 선명도
self.logger.info("[카메라 설정] 최적값 적용 완료")
```

### 웹 UI 기능

| 기능 | 설명 |
|-----|------|
| **실시간 비디오 스트림** | 카메라 영상과 마커 감지 상태 실시간 표시 |
| **현재 상태 패널** | 노출 모드, 현재 노출/게인 값, 밝기 통계 표시 |
| **마커 검출 통계** | 프레임 수, 검출 횟수, 검출률 (%) |
| **Auto/Manual 토글** | Auto Exposure ↔ Manual Exposure 전환 |
| **슬라이더 제어** | 실시간으로 노출/게인/밝기/대비/선명도 조정 |
| **리셋 버튼** | 모든 값을 기본값(50, 50, 4, 4, 4)으로 초기화 |
| **설정 저장** | 현재 설정을 터미널 로그에 출력 |
| **종료 버튼** | 디버깅 세션 종료 |


