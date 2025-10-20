# AI 면접 프로젝트 — 클라이언트 

>Streamlit 클라이언트는 면접 시나리오(질문 재생 → 답변 녹음 → 답변 자동 다운로드 → STT → LLM 피드백 → 자세/음정 분석)의 전체 플로우를 프론트에서 제어합니다. 서버는 녹음 시작/종료와 결과 파일 제공을 담당합니다.

## 주요 기능

* **면접 진행 UI**: 면접 시작, 답변 시작, 답변 종료, 면접 종료 
* **녹음 제어**: 서버에 `/command/start_record`, `/command/stop_record` 호출
* **결과 자동 다운로드**: WAV, XML(자세/표정), mp4
* **STT/피드백**: Whisper STT 및 LLM(Gemini 2.5 Flash) 피드백 체인
* **음성 안정성(eGeMAPS)**: opensmile로 jitter/shimmer/HNR 등 추출
* **총평과 항목별 피드백**: 답변별 + 최종 리포트



##

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
streamlit>=1.37
requests>=2.31
python-dotenv>=1.0
pydub>=0.25
opensmile>=2.5
numpy>=1.26
pandas>=2.0
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
RECORD_COUNTDOWN=10   # 녹음 전 카운트다운(초)
SESSION_PREFIX=default_session
```

## 디렉터리 구조

```text
AI-interview/
 ├─ app/
 │   ├─ main_app.py            # Streamlit 엔트리
 │   ├─ upload_section.py      # 업로드 섹션  
 │   ├─ interviewer.py         # 메인 섹션        
 │   ├─ adapters/
 │   │   ├─ interviewer_adapters.py  # 백엔드 연결 
 │   │   └─ posture_adapters.py          # xml -> llm 연결 브릿지
 │   ├─ assets/ # 면접관 동영상 저장 디렉터리 
 │
 ├─ core/
 │   ├─ analysis_audio.py # opensmile 음정분석 
 │   ├─ analysis_pose.py  # 자세 분석 라벨 
 │   ├─ analysis_pose_jetson.py # jetson 자세 분석 라벨 
 │   ├─ chains.py # 랭체인 프롬프트 
 │   ├─ recording_io.py # 파일 저장 유틸
 │   ├─ whisper_run.py # whisper STT 구현 
 ├─ requirements.txt
 └─ README.md (본 파일) 

```
## 데이터/결과물 구조

다운로드는 세션 폴더 기준으로 관리됩니다.

```text
./data/records
└─ <session_id>/
├─ audio_YYYYmmdd_HHMMSS.wav
├─ video_YYYYmmdd_HHMMSS.mp4
├─ log_YYYYmmdd_HHMMSS.xml

```

## 실행 방법

```bash
# 환경변수 로드 후
GOOGLE_API_KEY="사용할 구글 API KEY" streamlit run app/main_app.py
```

## 클라이언트 동작 흐름

1. **질문 재생** → 2)  **start_record 요청** → 3) **stop_record 요청** → 4) **WAV/분석 XML/mp4 다운로드** → 5) **STT** → 6) **해당 답변에 대한LLM 피드백 생성** → 7) **면접 종료 후 자세/표정 분석까지 들어간 최종 리포트 표시**

## 사용 REST API
* `POST /command/start_record`

  * 본문: 없음 (또는 JSON 옵션)
  * 응답: `200 OK`

* `POST /command/stop_record`

  * 본문: 없음
  * 응답: `200 OK` (파일이 저장되었음)

* `GET /download/wav/audio.wav`

  * 헤더: `Content-Type: audio/*`
  * 응답: WAV 바이트 스트림

* (예) `GET /download/xml/posture.xml`, `GET /download/xml/face.xml`

  * 헤더: `Content-Type: application/xml`




## 향후 계획

* 질문 셔플/난이도 다양하게 추가 
* 리포트 내 지표 임계값/정규화(마이크·환경 보정)
* 테스트 자동화(e2e): mock 서버로 다운로드/분석 파이프 검증
