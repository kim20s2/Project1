# Client(Streamlit) - Install & Run Guide

>Streamlit 클라이언트는 면접 시나리오(질문 재생 → 답변 녹음 → 답변 자동 다운로드 → STT → LLM 피드백 → 자세/음정 분석)의 전체 플로우를 프론트에서 제어합니다. 서버는 녹음 시작/종료와 결과 파일 제공을 담당합니다.

## 🛠 설치 방법 (Installation)
### 요구 사항

* Python 3.10
* OS: Ubuntu 24.04

### 설치

```bash
# 1) 가상환경
python3 -m venv .venv
source ./.venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 환경변수(.env) 준비 후 실행
cp .env.example .env && vi .env
```

### requirements.txt 
```txt
# =========================
# AI Interview Client - requirements.txt
# Python 3.10 권장
# =========================

# --- UI / App ---
streamlit==1.39.0
requests>=2.31,<3

# --- Data / Utils ---
pandas>=2.2,<2.3
numpy>=1.26,<3.0
python-dateutil>=2.9.0.post0
packaging>=24.0

# --- Audio I/O & Analysis ---
# openSMILE 파이썬 바인딩 (eGeMAPS functionals 사용)
opensmile==2.5.0
# WAV 입출력
soundfile>=0.12,<1.0
scipy>=1.11,<2

# --- XML 파싱(자세/표정 요약 XML) ---
xmltodict>=0.13,<0.14

# --- LLM (LangChain + Google Gemini) ---
# LLMChain, ChatPromptTemplate 사용
langchain==0.2.16
langchain-community>=0.2.0
# Gemini 연동
langchain-google-genai>=2.0.0
google-generativeai>=0.7.0

# --- 이미지/미디어(경량) ---
Pillow>=10.2,<12
imageio>=2.34,<3
imageio-ffmpeg>=0.4.9

# =========================
# 선택(옵션) 패키지
# 필요할 때만 주석 해제해서 사용
# =========================

# --- 로컬 Whisper STT ---
# 주의: torch는 OS/하드웨어마다 다른 휠이 필요함(아래 안내 참고)
# openai-whisper
# torch

# --- OpenCV (프레임 기반 표정/포즈 후처리 시) ---
# opencv-python>=4.9.0.80

# --- .env로 키 관리할 경우 ---
# python-dotenv>=1.0

# --- 개발 편의(코드 포맷터/린터) ---
# black
# ruff

# =========================
# 설치 팁 (주석)
# =========================
# 1) Whisper를 쓸 때는 ffmpeg 바이너리가 시스템에 필요합니다.
#    Ubuntu: sudo apt-get install -y ffmpeg
#    macOS (brew): brew install ffmpeg
#
# 2) torch는 플랫폼별로 다릅니다. 실패하면 아래처럼 별도 설치:
#    - Linux CUDA 12.x:  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
#    - Linux CPU-only:  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
#    - macOS(Apple Silicon, MPS): pip install torch==2.4.0
#
# 3) langchain 0.2.x에서는 일부 모듈이 langchain-community로 분리되었습니다.
#    (예: 일부 로더/커넥터) 필요 시 langchain-community를 함께 둡니다.
```

## 환경변수

`.env`에 설정합니다.

```env
# 서버 베이스 URL (예: 내부망 IP)
SERVER_URL=http://IP 주소

# LLM 피드백 체인
GOOGLE_API_KEY=YOUR_KEY
MODEL_NAME=gemini-2.5-flash
TEMPERATURE=0.6

# 다운로드/세션
DOWNLOAD_DIR=./data
SESSION_PREFIX=default_session
```


## 클라이언트 실행 방법

```bash
# 환경변수 로드 후
GOOGLE_API_KEY="사용할 구글 API KEY" streamlit run app/main_app.py
```

## 클라이언트 동작 흐름

1. **질문 재생** → 2)  **start_record 요청** → 3) **stop_record 요청** → 4) **WAV/분석 XML/mp4 다운로드** → 5) **STT** → 6) **해당 답변에 대한 LLM 피드백 생성** → 7) **면접 종료 후 자세/표정 분석까지 들어간 최종 리포트 표시**



