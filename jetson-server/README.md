# Jetson Nano FER+Posture Recorder — Install & Run Guide

이 문서는 **`run_3cls_audio2.py` + Flask 서버(`server.py`)**를 젯슨나노에서 실행하기 위해 필요한 요구사항, 설치 순서, 실행 방법, 트러블슈팅을 한 번에 정리한 **README.md**입니다.

---

## 1) 개요

- **기능**: 카메라 입력에서 **자세(포즈) + 3클래스 표정(negative/neutral/positive)**를 실시간 추론하고, **오버레이 영상, 원본 영상, 오디오, 이벤트 XML**을 저장합니다. 녹화는 키보드로 토글하며, Flask 서버를 통해 **시작/정지/다운로드/로그확인**을 HTTP API로 제어할 수 있습니다.

- **고정 경로(엔진 파일)**  
  - 얼굴감정: `/home/jetson10/fer/scripts/face_3cls.engine`  
  - 포즈: `/home/jetson10/fer/scripts/pose_movenet.engine`

- **출력 폴더(기본)**: `/home/jetson10/fer/{mp4,wav,xml}` (서버 구동시 `/home/jetson10/fer/server/srv_tmp/{mp4,wav,xml}`)

---

## 2) 사전 요구사항

- **Jetson Nano (JetPack 설치, TensorRT 포함)**  
  - TensorRT Python 바인딩: `python3-libnvinfer`, `python3-libnvinfer-dev` (JetPack 기본 제공)  
- **X 환경(로컬 모니터/VNC 등)**: 서버의 자동 키 입력(xdotool)이 **보이는 창**에 전송되어야 합니다. (headless 콘솔만으로는 불가)
- **카메라/마이크**가 연결되어 있어야 합니다. (기본 마이크 ALSA 디바이스: `plughw:2,0`)

---

## 3) 설치 순서

### A. 시스템 패키지 설치 (APT)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-dev build-essential git \
  libopencv-dev python3-opencv libopencv-data \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  xdotool x11-xserver-utils alsa-utils ffmpeg \
  python3-libnvinfer python3-libnvinfer-dev
```

- `python3-opencv`, `libopencv-data`: **Haar Cascade** 파일 제공(얼굴 폴백 검출)  
- `xdotool`, `x11-xserver-utils`: 서버에서 **키 자동전송**(b/t/q)과 **auto-repeat on/off**에 필요  
- `alsa-utils`: **오디오 녹음(arecord)**에 필요  
- `gstreamer*`, `ffmpeg`: 코덱/비디오 입출력 호환성 향상 (OpenCV VideoWriter `mp4v` 사용)  
- `python3-libnvinfer*`: TensorRT Python 바인딩 (엔진 실행)

### B. 파이썬 패키지 설치 (PIP)

> **권장**: 시스템 파이썬(젯팩) 그대로 사용. (별도 venv 선택 가능)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install flask pycuda numpy
```

- `tensorrt`는 JetPack에서 apt로 제공되므로 pip 설치 **불필요**. 코드상 TRT는 `tensorrt` 모듈과 PyCUDA를 직접 사용합니다.

### C. 디렉터리/파일 배치

```bash
# 예시 위치 (코드 상 고정 경로 기준)
mkdir -p /home/jetson10/fer/scripts
mkdir -p /home/jetson10/fer/server

# 업로드한 파일 배치
# /home/jetson10/fer/scripts/run_3cls_audio2.py
# /home/jetson10/fer/scripts/trt_utils.py
# /home/jetson10/fer/server/process_call.py
# /home/jetson10/fer/server/server.py

# 엔진 파일(필수)
# /home/jetson10/fer/scripts/face_3cls.engine
# /home/jetson10/fer/scripts/pose_movenet.engine
```

- 엔진 경로는 코드에 **하드코딩**되어 있으니 정확히 맞춰두세요.  
- HaarCascade는 보통 `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`에 설치됩니다. 코드가 여러 경로를 자동 탐색합니다.

---

## 4) 단독 실행(run_3cls_audio2.py)

### 기본 실행

```bash
cd /home/jetson10/fer/scripts
python3 run_3cls_audio2.py --src 3 --width 640 --height 480 --fps 30 --show_fps
```

- 주요 기본값  
  - `--src`: 기본 `"3"` (USB/Webcam 번호 또는 파일 경로)  
  - 출력 폴더: `/home/jetson10/fer/{mp4,wav,xml}` 자동 생성  
  - 코덱: `--rec_codec mp4v` (기본)  
  - 마이크: `--mic_device plughw:2,0`, `--mic_rate 16000`, `--mic_channels 1`, `--mic_format S16_LE`

### 키 조작

- **`b`**: 자세 **baseline 캡처**(이후 이벤트 로깅 활성)  
- **`r`**: baseline 해제(로깅 비활성)  
- **`t`**: **녹화 토글** (시작/정지)  
- **`o`**: 90° 회전  
- **`q` 또는 `Esc`**: 종료

### 출력물

- **오버레이 영상**: `mp4/rec_YYYYmmdd_HHMMSS_annot.mp4`  
- **원본 영상**: `mp4/rec_YYYYmmdd_HHMMSS_raw.mp4`  
- **오디오**: `wav/rec_YYYYmmdd_HHMMSS.wav`  
- **이벤트 XML**: `xml/xml_YYYYmmdd_HHMMSS.xml` (baseline ON + 녹화 중 조건 만족 시)

> **참고**: `--out_dir`를 지정하면 파일명이 고정(`video_ai.mp4`, `video.mp4`, `audio.wav`, `log.xml`)으로 저장됩니다. 서버는 이 모드를 사용합니다.

---

## 5) 서버 모드(Flask)

서버는 `xdotool`을 사용해 GUI 창에 **b → t** 키를 자동 전송하여 baseline/녹화를 시작하고, 정지 시 **t → q** 순서로 종료해 파일 안정화를 보장합니다. 실행 중 로그를 파일로 남깁니다.

### 실행

```bash
cd /home/jetson10/fer/server
python3 server.py
# 기본 포트 5000, 0.0.0.0 바인딩
```

- 내부적으로 `ProcessManager`가 `run_3cls_audio2.py`를 **`--out_dir /home/jetson10/fer/server/srv_tmp`**로 강제 실행합니다. (서버 전용 작업폴더)

### REST API

- **녹화 시작**  
  `GET /command/start_record?src=1`  
  - 창이 보이면: **b → t** 자동키 전송(각 1회) 후 녹화 시작, 로그 경로 반환
- **녹화 정지**  
  `GET /command/stop_record`  
  - **t**로 녹화 종료 → 파일 **생성/사이즈 안정 대기** → **q**로 창 종료
- **상태 조회**  
  `GET /api/status` → 실행 중 여부, 최신 파일 리스트(mp4/wav/xml), 마지막 실행 커맨드/로그 경로 반환
- **파일 다운로드**  
  `GET /download/<kind>/<filename>` (`kind`: mp4|wav|xml)
- **로그 꼬리 보기**  
  `GET /debug/log` → 마지막 로그 파일 tail 반환

### cURL 예시

```bash
# 시작 (카메라 /dev/video1 가정)
curl "http://<JETSON_IP>:5000/command/start_record?src=1"

# 상태
curl "http://<JETSON_IP>:5000/api/status"

# 다운로드 (예: 최근 mp4 파일명 확인 후)
curl -OJ "http://<JETSON_IP>:5000/download/mp4/<filename>.mp4"

# 정지
curl "http://<JETSON_IP>:5000/command/stop_record"

# 로그
curl "http://<JETSON_IP>:5000/debug/log"
```

---

## 6) 자주 쓰는 실행 옵션

- **해상도/FPS 조정**: `--width 640 --height 480 --fps 30`  
- **퍼포먼스**  
  - `--max_side 720` (긴 변 리사이즈)  
  - `--frame_skip N` (프레임 스킵)  
  - `--process_every N` (파일 입력 시 grab 후 N프레임마다 처리)
- **녹화 모드**  
  - `--rec_mode auto|realtime|fixed` + `--rec_fps` (fixed일 때만 고정 FPS)
- **표정 튜닝**  
  - `--fer_gamma 1.2`, `--fer_clahe`, `--fer_tta_flip`, `--fer_temp 1.2`, `--fer_labels "neg,neu,pos"`
- **마이크**  
  - `--mic_device plughw:2,0` (또는 `arecord -l`로 카드/디바이스 확인 후 변경)

---

## 7) 트러블슈팅

- **창이 안 뜨거나 키 입력이 안 먹음**  
  - 로컬 X 세션/데스크톱 또는 VNC 위에서 실행해야 합니다. `xdotool`은 **보이는 창**에만 키를 보냅니다. (SSH 터미널 전용은 불가)  
  - `echo $DISPLAY` 확인, VNC 사용 시 해당 세션에서 서버 실행.

- **마이크 녹음 실패**  
  - `arecord -l`로 장치 번호 확인 후 `--mic_device`를 `plughw:<card>,<device>` 형태로 지정하세요.

- **엔진 로딩 실패**  
  - 경로/파일명 확인: `/home/jetson10/fer/scripts/face_3cls.engine`, `pose_movenet.engine`  
  - TensorRT/pycuda 설치 확인. (JetPack의 `python3-libnvinfer*`, `pycuda`)

- **얼굴 검출이 자주 실패**  
  - 포즈로 얼굴박스를 추정하고, 실패 시 HaarCascade로 폴백합니다. 조명/해상도/거리 조정 및 `--pose_thresh` 튜닝을 고려하세요.

- **MP4 저장 문제**  
  - `--rec_codec mp4v`가 기본입니다. 필요 시 `ffmpeg`, `gstreamer` 플러그인을 추가 설치했습니다. 해상도/프레임/코덱 호환 이슈가 있으면 다른 코덱(`XVID` 등) 시도.

---

## 8) 성능 팁

- 입력 해상도를 낮추거나 `--max_side`로 리사이즈.  
- `--frame_skip`/`--process_every`로 처리 부하 감소.  
- 밝기/감마(`--fer_gamma`)와 CLAHE(`--fer_clahe`)로 FER 안정화.

---

## 9) 파일 구조(예시)

```
/home/jetson10/fer
├── mp4/                     # 단독 실행 기본 출력
├── wav/
├── xml/
├── scripts/
│   ├── run_3cls_audio2.py   # 메인 실행 스크립트
│   ├── trt_utils.py         # TensorRT 래퍼
│   ├── face_3cls.engine     # FER 엔진 (필수)
│   └── pose_movenet.engine  # MoveNet 엔진 (필수)
└── server/
    ├── server.py            # Flask 서버
    ├── process_call.py      # 실행/키전송/로그/파일안정화 관리
    └── srv_tmp/
        ├── mp4/             # 서버 실행시 출력 폴더(고정명)
        ├── wav/
        └── xml/
```

---

## 10) 서버 아키텍처 & 설계 (간단 요약)

- **웹 레이어**: Flask HTTP API (`server.py`)  
  - `/command/start_record` → 프로세스 시작 + **b→t** 키 전송으로 baseline/녹화 진입  
  - `/command/stop_record` → **t**(녹화종료) 대기 → **q**(창종료)  
  - `/api/status` 파일 최신목록/상태 제공, `/download/*` 파일 다운로드, `/debug/log` 로그 tail

- **프로세스 제어**: `ProcessManager` (`process_call.py`)  
  - `xdotool`/`xset`를 통해 키 전송 & auto-repeat 제어  
  - 로그 파일 기록, **출력 파일 사이즈 안정화** 후 종료 처리

- **추론 코어**: `run_3cls_audio2.py`  
  - TensorRT 엔진 로드(`TRTModule`) → 포즈(MoveNet) + FER(3cls) 추론  
  - baseline 대비 자세 스코어링, 다리 떨림 FFT 감지, 이벤트 XML 기록  
  - 오디오(arecord) 병렬 녹음 & 오버레이/원본 영상 동시 저장

---

## 11) 빠른 체크리스트

- [ ] 엔진 파일 2개를 정확한 경로에 배치했다.  
- [ ] `python3-libnvinfer*`, `pycuda`, `flask`, `numpy`, `opencv` 설치 완료.  
- [ ] X 세션에서 실행 중이며 `xdotool`이 동작한다.  
- [ ] 마이크 ALSA 디바이스를 확인했다(필요시 `--mic_device` 수정).
