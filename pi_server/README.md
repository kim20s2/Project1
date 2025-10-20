# pi\_server

`pi_server`는 라즈베리파이 5와 Hailo-8 AI 가속기를 사용하여 사용자의 데이터를 실시간으로 분석하는 RESTful API 서버입니다.

카메라와 마이크 입력을 받아 AI 모델을 통해 사용자의 포즈(자세 교정)와 표정(웃음 여부)을 실시간으로 인식하며, 클라이언트의 요청에 따라 분석 결과를 전송합니다.

<center>
<img src="../assets/images/pi_webcam.jpeg" alt="Image" width="" height="200">
<img src="../assets/images/pi_a6700.jpeg" alt="Image" width="" height="200">
</center>

## 🛠 설치 방법 (Installation)

### 1\. Hailo-8 하드웨어 및 펌웨어 설정

먼저 [Raspberry Pi 5 및 Hailo 설정 공식 가이드](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/install-raspberry-pi5.md#how-to-set-up-raspberry-pi-5-and-hailo)를 따라 라즈베리파이 5와 Hailo-8을 설정합니다.

### 2\. HailoRT 버전 확인

  * 위 가이드를 따르면 기본 Python 환경에 `hailoRT 4.20` 버전이 설치됩니다.
  * **⚠️ 중요:** 이후 `hailoRT` 버전이 업데이트되더라도, 이 프로젝트는 **`4.20` 버전에 맞게 테스트되었습니다.** `4.20` 버전으로 설치하는 것을 강력히 권장합니다.

### 3\. 가상 환경 확인

`hailoRT` 설치 후, 프로젝트 폴더 내의 Python 가상 환경(`env`)이 정상적으로 실행되는지 확인합니다.

## 🚀 사용 방법 (Usage)

### 1\. 가상 환경 활성화

```bash
source env.sh
```

### 2\. 서버 실행

사용할 장치(device)를 지정하여 서버를 실행합니다.

```bash
bash server.sh [DEVICE_NAME]
```

**지원 장치 (`DEVICE_NAME`)**

  * `webcam`: 일반 웹캠 (640x480 @ 30fps 기준)
  * `file`: 로컬 동영상 파일 재생
  * `a6700`: Sony a6700 (1280x720 @ 30fps 기준)

## 🧪 테스트 스크립트 용도

서버의 개별 기능을 테스트하기 위한 셸 스크립트입니다.

  * `run_rt.sh`: (Routing) 하나의 영상 소스에서 각 프로세스(fa, po, re)로 영상 데이터를 분배합니다. **다른 세 개의 스크립트(`run_fa.sh`, `run_po.sh`, `run_re.sh`)를 실행하기 전에 반드시 먼저 실행되어야 합니다.**
  * `run_fa.sh`: (Facial Analysis) 얼굴 표정 인식 프로세스를 실행하고 결과를 `xml`로 저장합니다.
  * `run_po.sh`: (Pose Estimation) 자세 인식 프로세스를 실행하고 결과를 `xml`로 저장합니다.
  * `run_re.sh`: (Recording) 녹화 프로세스를 실행합니다.

### 예시

```bash
# cmd 1
bash run_rt.sh [DEVICE_NAME]

# cmd 2
bash run_po.sh
```
  * [DEVICE_NAME]은 서버 실행 시와 같습니다.
  * 제공된 가상환경의 사용을 전제합니다.

### TODO

#### 추후 추가사항

* 사용법: 서버가 켜지는 과정을 보여주는 터미널 녹화 GIF