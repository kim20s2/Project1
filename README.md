# 오만과 편견 (Pride and Prejudice)

***AI 면접관의 '편견'과 사용자의 '오만(무의식적 습관)'을 극복하는 가상 면접 피드백 시스템***

본 프로젝트는 면접을 준비하는 사용자에게 AI 기반의 종합적인 피드백을 제공하는 시스템입니다.

사용자의 시각적 요소(자세, 표정)와 청각적 요소(목소리 톤, 답변 내용)를 실시간으로 분석하고, 면접 종료 후 Google Gemini (LLM)를 통해 개선점을 담은 상세한 리포트를 제공합니다.

## 🏛️ 시스템 아키텍처 (System Architecture)

이 시스템은 실시간 데이터 수집 및 AI 추론을 담당하는 **Edge AI 서버**와, 데이터를 취합하여 사용자에게 피드백을 생성/제공하는 **클라이언트 애플리케이션**으로 구성됩니다.

<center>
<img src="./assets/images/server&client1.png" alt="Image" width="350" height="350">
</center>

### 1. Edge AI 서버 (Server)

* **역할:** 면접자의 영상과 음성을 실시간으로 수집하고 AI 모델을 통해 분석합니다.
* **하드웨어:** Raspberry Pi 5 (+ Hailo-8) 또는 Jetson Nano
* **주요 기능:**
    * GStreamer와 RTSP를 통해 카메라 입력을 받아 면접자의 **얼굴 표정**(예: 웃음 여부)과 **신체 자세**(예: 바른 자세 여부)를 실시간으로 추론합니다. (TensorFlow, OpenCV, HailoRT 활용)
    * 분석된 데이터를 시간대별로 태깅하여 `xml` 특징 파일로 생성합니다.
    * Flask 기반의 RESTful API를 통해 클라이언트의 요청 시 녹화된 영상/음성 원본과 `xml` 파일을 전송합니다.

### 2. 클라이언트 (Client)

* **역할:** 서버로부터 데이터를 받아 사용자에게 보여주고, LLM을 통해 최종 피드백을 생성합니다.
* **주요 기능:**
    * Streamlit 기반의 UI를 통해 사용자로부터 면접 시작/종료 명령을 받습니다.
    * 서버에 녹화 결과물(영상, 음성, `xml`)을 요청하고 다운로드합니다.
    * **음성 분석:**
        * `Whisper`를 사용해 음성 파일을 텍스트(답변 내용)로 변환합니다.
        * `OpenSmile`을 사용해 목소리 떨림, 톤, 말의 빠르기 등 음향적 특징을 추출합니다.
    * **LLM 피드백 생성:**
        * `Langchain`을 사용하여 [서버의 `xml` 데이터 (자세, 표정)] + [클라이언트의 분석 데이터 (답변 내용, 목소리 특징)]를 종합합니다.
        * 모든 데이터를 Google Gemini (LLM)에 전송하여 "면접관"의 관점에서 상세한 피드백 리포트를 생성합니다.

## ⚙️ 작동 흐름 (Workflow)

<center>
<img src="./assets/graphs/RESTful_flow.svg" alt="Image" width="525" height="525">
</center>

1.  **(Client)** 사용자가 Streamlit 앱에서 "면접 시작" 버튼을 누릅니다.
2.  **(Server)** 클라이언트의 요청을 받아 카메라/마이크 녹화를 시작함과 동시에 AI 추론(자세, 표정)을 시작합니다.
3.  **(Client)** 사용자가 "면접 종료" 버튼을 누릅니다.
4.  **(Server)** 녹화를 중지하고, 수집된 특징 데이터를 `xml` 파일로 최종 정리합니다.
5.  **(Client)** 서버에 결과물을 요청하여 영상, 음성, `xml` 파일을 다운로드합니다.
6.  **(Client)** `Whisper`와 `OpenSmile`로 음성 파일을 분석하고, `xml` 파일을 파싱합니다.
7.  **(Client)** `Langchain`을 통해 모든 데이터를 취합하여 Google Gemini API에 전송합니다.
8.  **(Client)** LLM이 생성한 피드백 리포트(개선점, 칭찬 등)를 사용자에게 보여줍니다.

## 💻 기술 스택 (Tech Stack)

### Server (Edge AI)

* **Language:** Python
* **AI/ML:** HailoRT, TensorFlow, OpenCV
* **Media:** GStreamer, RTSP
* **API:** Flask

### Client (Feedback Generation)

* **Language:** Python
* **Framework:** Streamlit
* **LLM Orchestration:** Langchain
* **Audio Analysis:** Whisper (STT), OpenSmile (Acoustic Features)

### LLM

* Google Gemini (via Google AI Studio)

## 🚀 설치 및 실행 방법 (Setup & Usage)

### 1. 서버 (Server)

* 서버(Raspberry Pi / Jetson Nano) 설정 및 실행 방법은 아래 링크를 참조하세요.
* [Raspberry Pi 서버 설치 가이드 (Raspberry Pi Server Setup Guide)](./pi_server/README.md)
* [Jetson Nano 서버 설치 가이드 (Jetson Nano Server Setup Guide) - 링크 추가 예정](./jetson_server/README.md)

### 2. 클라이언트 (Client)

* 클라이언트 애플리케이션 실행 방법은 아래 링크를 참조하세요.
* [클라이언트 실행 가이드 (Client Usage Guide) - 링크 추가 예정](./client/README.md)

### TODO


* 시스템 아키텍처 바로 위: 최종 결과물 데모 

* 시스템 아키텍처 내부: AI 분석 과정 시연 