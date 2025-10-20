import streamlit as st
import streamlit.components.v1 as components
import time, re
from pathlib import Path
import os
import numpy as np
import librosa
import scipy.ndimage
import tempfile
import requests

# ============================
# 0️⃣ API 키 설정
# ============================
# Google GenAI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ============================
# 1️⃣ Whisper STT
# ============================
import whisper
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small")  # small/medium/large 선택

def run_whisper(audio_file):
    result = model.transcribe(audio_file, language="ko")
    return result["text"]
print("현재 장치:", device)
# ============================
# 2️⃣ 목소리 안정성 분석
# ============================
import opensmile
import pandas as pd

# 🔹 openSMILE 초기화
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def analyze_stability(audio_path: str):
    """음성 안정성 피처 추출"""
    result = smile.process_file(audio_path).iloc[0]

    features = {
        "jitter": result["jitterLocal_sma3nz_amean"],
        "shimmer": result["shimmerLocaldB_sma3nz_amean"],
        "hnr": result["HNRdBACF_sma3nz_amean"],
        "f0_std": result["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
        "loudness_std": result["loudness_sma3_stddevNorm"],
    }
    return features, "ok"

def get_stability_score(jitter, shimmer,hnr):
 # 1️⃣ 기준 완화
    JITTER_REF = 0.07
    SHIMMER_REF = 0.6

    # 2️⃣ 개별 점수
    score_jitter = max(0, 1 - (jitter / JITTER_REF))
    score_shimmer = max(0, 1 - (shimmer / SHIMMER_REF))
    score_hnr = min(1.0, max(0, hnr / 20.0))  # 20dB 이상이면 매우 깨끗함

    # 3️⃣ 가중 평균 (HNR 비중 ↑)
    score = (score_jitter * 0.25 + score_shimmer * 0.25 + score_hnr * 0.5) * 10

    # 4️⃣ 스케일 보정 (분포 확장)
    score = min(10.0, (score * 1.4) + 2.0)

    # 5️⃣ 라벨
    if score >= 8.0:
        label, color = "안정적 ✅", "success"
    elif score >= 5.0:
        label, color = "보통 ⚠️", "warning"
    else:
        label, color = "불안정 ❌", "error"

    return round(score, 2), label, color


# ============================
# 3️⃣ 자세 분석
# ============================
import xml.etree.ElementTree as ET

def parse_posture_summary(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    head_tilt_count = 0
    body_tilt_count = 0
    gesture_count = 0
    frame_total = 0

    for frame in root.findall("frame"):
        frame_total += 1
        analysis = frame.find("analysis")
        if analysis is None:
            continue

        for result in analysis.findall("result"):
            rtype = result.get("type", "")
            if rtype == "head_tilt":
                head_tilt_count += 1
            elif rtype == "body_tilt":
                body_tilt_count += 1
            elif rtype == "gesture":
                gesture_count += 1

    # 기본 라벨링
    labels = []
    if head_tilt_count > frame_total * 0.3:
        labels.append("머리 기울임 잦음")
    if body_tilt_count > frame_total * 0.3:
        labels.append("몸 기울어짐 잦음")

    # 제스처 라벨링
    if gesture_count == 0:
        gesture_label = "제스처 없음 (답변이 딱딱할 수 있음)"
    elif 1 <= gesture_count <= 3:
        gesture_label = "자연스러운 제스처 사용"
    else:
        gesture_label = "제스처 과다 (산만할 수 있음)"

    labels.append(gesture_label)

    return {
        "frames": frame_total,
        "head_tilt_count": head_tilt_count,
        "body_tilt_count": body_tilt_count,
        "gesture_count": gesture_count,
        "label": ", ".join(labels)
    }


# ============================
# 4️⃣ Google GenAI + LangChain
# ============================
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 전문 면접관이다. 아래는 면접자의 행동 분석 결과이다. 음성, 자세, 제스처 등의 데이터를 참고하여 면접자의 발성과 태도를 평가하고 피드백을 제공하라."),
    ("human", """
면접 답변:
- Transcript: {transcript}

[음성 분석 요약]
- 안정성 점수: {stability_score:.2f}/10
- 전반적 평가 등급: {label}

[자세 및 제스처]
- {posture}

요구사항:
1. 위 데이터를 참고해 면접자의 **발성 안정성, 자신감, 전달력, 태도**를 종합적으로 평가하라.  
2. Jitter, Shimmer, HNR 같은 기술적 용어는 언급하지 말고,  
   **청각적으로 느껴지는 인상(예: 떨림, 안정감, 자신감 등)** 으로 표현하라.  
3. 피드백은 **3문장 이내**로 구성하며,  
   - 첫 문장은 강점을 짚고  
   - 두 번째 문장은 개선점을 부드럽게 제시하며  
   - 세 번째 문장은 면접자의 성장 가능성을 긍정적으로 마무리하라.  
4. 말투는 **전문적이고 따뜻하게**, 실제 면접관처럼 작성하라.  

출력 형식:
- 분석 요약: ...
- 강점: ...
- 피드백: ...
""")
])

chain = LLMChain(llm=llm, prompt=prompt)


# ============================
# streamlit 실행
# ============================


st.set_page_config(page_title="AI 면접관", page_icon="🎤", layout="wide")

# 초기화(렌더 전에 1회)
ss = st.session_state
ss.setdefault("stopped", False)          # 종료 후에만 보일 섹션 제어
ss.setdefault("mp4_manual_only", False)    # 체크박스 기본: ON → MP4만 수동
ss.setdefault("auto_mp4", True)          # 전체 자동 on/off (필요 없다면 True로 고정해도 됨)
ss.setdefault("auto_saved_once", False)  # 새 녹화 시작 시 False로 리셋 필요 (start_record 직후 등)


# 타이틀 + 서브타이틀
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF; font-size: 40px;'>
        🎤 AI 면접관
    </h1>
    <h3 style='text-align: center; color: #555;'>
        당신의 <b>답변 · 자세 · 표정</b>을 종합 분석합니다
    </h3>
    """,
    unsafe_allow_html=True
)

st.divider()

# 라즈베리파이 Flask 서버 IP
SERVER_URL = "http://10.10.14.00:5000"

DL_DIR = Path("downloaded")
DL_DIR.mkdir(parents=True, exist_ok=True)

st.subheader("🎙️ 면접 시작")

# 세션ID 안전화(슬래시/공백 등 제거)
raw_session_id = ss.get("session_id", "default_session")
SESSION_ID = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(raw_session_id))

# 세션별 저장 폴더 (DL_DIR 재사용)
SAVE_DIR = DL_DIR / SESSION_ID
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def _wait_and_get(url, max_wait_s=8, interval_s=0.5):
    """GET만으로 준비 확인. 일부 서버의 HEAD/Content-Length 미지원을 회피."""
    deadline = time.monotonic() + max_wait_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with requests.get(url, timeout=10, stream=True) as r:
                if r.status_code == 200:
                    # Content-Length가 있으면 >0 확인, 없으면 첫 청크 존재 확인
                    clen = r.headers.get("Content-Length")
                    if (clen and int(clen) > 0) or any(r.iter_content(chunk_size=1024)):
                        # 다시 전체 바이트를 받아야 한다면 재요청(혹은 위에서 모은 청크를 저장)
                        return requests.get(url, timeout=10)  # 전체 바이트
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    if last_err:
        st.info(f"파일 확인 중: {last_err}")
    return None

def _save_local(resp, filename_hint):
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = Path(filename_hint)
    fname = f"{p.stem}_{SESSION_ID}_{ts}{p.suffix}"
    path = SAVE_DIR / fname
    path.write_bytes(resp.content)
    return path.resolve()

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ 면접 시작"):
        try:
            r = requests.get(f"{SERVER_URL}/command/start_record", timeout=5)
            r.raise_for_status()
            ss["recording"] = True
            ss["stopped"] = False           # ✅ 시작하면 반드시 False
            ss["auto_saved_once"] = False   # ✅ 자동저장 가드 리셋
            st.success("면접이 시작되었습니다.")
        except requests.exceptions.RequestException as e:
            st.error(f"면접 시작 실패: {e}")

with col2:
    if st.button("⏹️ 면접 종료"):
        try:
            r = requests.get(f"{SERVER_URL}/command/stop_record", timeout=5)
            r.raise_for_status()
            ss["recording"] = False
            ss["stopped"] = True            # ✅ 종료 후 True
            st.success("면접이 종료되었습니다.")
            st.rerun()                      # ✅ 바로 자동저장 블록 실행시키고 싶으면
        except requests.exceptions.RequestException as e:
            st.error(f"면접 종료 실패: {e}")

# 상태 표시
if 'recording' in st.session_state and st.session_state['recording']:
    st.markdown("🔴 **현재 면접 중입니다...**")
else:
    st.markdown("⚪ **대기 중**")

# 체크박스 (value= 빼서 경고 방지)
st.checkbox("MP4는 수동으로 받기", key="mp4_manual_only")
def wait_until_ready(url, max_wait_s=8, interval_s=0.5):
    """
    파일 준비 확인:
    1) HEAD 시도 (일부 서버 미지원)
    2) 실패/부정확 시 Range GET(0-0)로 존재만 확인
    준비되면 최종 GET Response 반환, 아니면 None
    """
    deadline = time.time() + max_wait_s
    last_err = None

    while time.time() < deadline:
        try:
            # 1) HEAD
            h = requests.head(url, timeout=2)
            if h.status_code == 200 and int(h.headers.get("Content-Length", "0")) > 0:
                g = requests.get(url, timeout=10, allow_redirects=True)
                if g.ok and g.content:
                    return g

            # 2) Range GET fallback (0-0 바이트만)
            r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=3, allow_redirects=True, stream=True)
            # 206 Partial Content 또는 200 OK(범위 무시)면 존재 판단
            if r.status_code in (200, 206):
                g = requests.get(url, timeout=10, allow_redirects=True)
                if g.ok and g.content:
                    return g

        except Exception as e:
            last_err = e

        time.sleep(interval_s)

    if last_err:
        st.info(f"파일 확인 중 오류: {last_err}")
    return None



# ------------------ 여기부터 stop_record 성공 직후 블록에 붙이기 ------------------
FILES = {
    "mp4": {"url": f"{SERVER_URL}/download/mp4/video.mp4", "mime": "video/mp4",       "name": "video.mp4"},
    "wav": {"url": f"{SERVER_URL}/download/wav/audio.wav", "mime": "audio/wav",       "name": "audio.wav"},
    "xml": {"url": f"{SERVER_URL}/download/xml/log.xml",   "mime": "application/xml", "name": "log.xml"},
}

# 자동 대상 결정
order = ["mp4", "wav", "xml"]
auto_all = ss["auto_mp4"]
auto_kinds = [k for k in order if not (k == "mp4" and ss["mp4_manual_only"])]
# => OFF이면 ["mp4","wav","xml"], ON이면 ["wav","xml"]

ready = {}

# ───────────── 자동 처리(단, mp4_manual_only면 MP4는 제외) ─────────────
if ss.get("stopped", False) and auto_all and auto_kinds and not ss["auto_saved_once"]:
    ready = {}  # ← 매번 초기화 (중요)
    with st.spinner("파일 준비 확인 중..."):
        for kind in auto_kinds:
            info = FILES[kind]
            res = wait_until_ready(info["url"], max_wait_s=8, interval_s=0.5)
            if res and res.content:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"{Path(info['name']).stem}_{SESSION_ID}_{ts}{Path(info['name']).suffix}"
                save_path = (SAVE_DIR / fname)
                save_path.write_bytes(res.content)
                ready[kind] = {"url": info["url"], "mime": info["mime"], "save": save_path}

    if ready:
        ss["auto_saved_once"] = True
        st.success("자동 저장 완료! (로컬 디스크)")
        for kind in ["mp4", "wav", "xml"]:
            if kind in ready:
                st.write(f"✅ `{kind}` 저장: `{ready[kind]['save'].resolve()}`")
    else:
        st.info("파일이 아직 준비되지 않았습니다.")
# ───────────── MP4 수동 다운로드 섹션 ─────────────
# mp4_manual_only=True 이거나, auto_all=False 인 경우 수동 버튼 제공
if st.session_state.get("mp4_manual_only", True) or not auto_all:
    info = FILES["mp4"]
    # 필요 시 준비 대기 후 받기
    if st.button("⬇️ MP4 받기"):
        res = wait_until_ready(info["url"], max_wait_s=8, interval_s=0.5)
        if res and res.content:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{Path(info['name']).stem}_{SESSION_ID}_{ts}{Path(info['name']).suffix}"
            (SAVE_DIR / fname).write_bytes(res.content)
            st.success("MP4 다운로드 준비 완료!")
            st.download_button(
                "파일 저장(브라우저)",
                data=res.content,
                file_name=fname,
                mime=info["mime"],
                use_container_width=True
            )
            st.caption(f"로컬 저장: `{(SAVE_DIR / fname).resolve()}`")
        else:
            st.warning("MP4가 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요.")
            
with st.expander("📤 업로드 모드", expanded=False):
    # ----------------------
    # 세션 상태 초기화
    # ----------------------
    if "chapters" not in st.session_state:
        st.session_state.chapters = []  # 전체 면접 기록 (세션별)
    if "history" not in st.session_state:
        st.session_state.history = []   # 현재 세션 기록

    # ----------------------
    # 새 면접 세션 시작 버튼
    # ----------------------
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "posture_key" not in st.session_state:
        st.session_state.posture_key = 0

    if st.button("🆕 새로운 면접 시작"):
        if st.session_state.history:
            st.session_state.chapters.append(st.session_state.history)
        st.session_state.history = []
        st.session_state.uploader_key += 1  # key 바꿔줌
        st.session_state.posture_key += 1
        st.rerun()


    # 서비스 소개 카드
    st.markdown(
        """
        <div style="
            background-color:#f0f8ff;
            padding:20px;
            border-radius:12px;
            text-align:center;
            border: 1px solid #1E90FF;
            margin-bottom:15px;
        ">
            <h3 style="color:#1E90FF;">🎙️ 면접 답변 업로드</h3>
            <p style="color:#333;">지원자의 음성과 자세를 업로드해주세요.<br>
            지원 형식: <b>WAV, M4A, MP3, FLAC, XML</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "🎙️ 음성 데이터",
        type=["wav","m4a","mp3","flac"],
        key=f"file_uploader_{st.session_state.uploader_key}"  # key 매번 바뀜
    )

    posture_file = st.file_uploader(
        "🧍 자세 데이터",
        type=["xml"],
        key=f"posture_uploader_{st.session_state.posture_key}" # key 매번 바뀜
    )
    summary = None
    if posture_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
            tmp_xml.write(posture_file.getbuffer())
            summary = parse_posture_summary(tmp_xml.name)  # 자세 분석 함수 사용

    if uploaded_file and posture_file:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name, posture_file.name}")
        st.audio(uploaded_file)
        # 임시 파일로 저장 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        with st.spinner("분석 중..."):
            # 1️⃣ 음성 텍스트 변환
            transcript = run_whisper(tmp_path)

            # 2️⃣ openSMILE 기반 피처 추출
            features, _ = analyze_stability(tmp_path)
            jitter = features["jitter"]
            shimmer = features["shimmer"]

            # 3️⃣ 안정성 점수 계산
            stability_score, voice_label, color = get_stability_score(
            jitter=features["jitter"],
            shimmer=features["shimmer"],
            hnr=features["hnr"]
        )

            # 4️⃣ LLM 피드백 생성
            feedback = chain.run({
            "transcript": transcript,
            "stability_score": stability_score,
            "label": voice_label,
            "posture": summary["label"] if summary else "데이터 없음"
        })
            

            # 5️⃣ 결과 저장
            result = {
                "transcript": transcript,
                "jitter": jitter,
                "shimmer": shimmer,
                "stability_score": stability_score,
                "voice_label": voice_label,
                "color": color,
                "posture": summary["label"] if summary else "데이터 없음",
                "feedback": feedback
            }

            st.session_state.history.append(result)


    def render_question_result(i, res):
        st.markdown(f"### ❓ 답변 {i}")
        st.info(res["transcript"])

        col1, col2 = st.columns([1, 2])
        with col1:
            # openSMILE 기반 주요 피처 표시
            st.metric("🎚️ Jitter (피치 흔들림)", f"{res['jitter']:.4f}")
            st.metric("🔉 Shimmer (볼륨 흔들림)", f"{res['shimmer']:.4f}")
        with col2:
            # 안정성 점수 시각화 (10점 환산)
            score = float(res.get("stability_score", 0))
            st.metric("목소리 안정성 점수", f"{score:.2f}/10")
            st.progress(score / 10)

        # 안정성 라벨 색상 표시
        label = res.get("voice_label", "데이터 없음")
        color = res.get("color", "warning")

        if color == "success":
            st.success(f"✅ 목소리 안정성: {label}")
        elif color == "warning":
            st.warning(f"⚠️ 목소리 안정성: {label}")
        else:
            st.error(f"❌ 목소리 안정성: {label}")

        if "posture" in res:
            if "안정적" in res["posture"]:
                st.success(f" 자세: {res['posture']}")
            elif "불안정" in res["posture"] or "기울어짐" in res["posture"]:
                st.error(f" 자세: {res['posture']}")
            else:
                st.warning(f" 자세: {res['posture']}")

        st.success(f"{res['feedback']}")
        st.divider()


    # ---------------- 결과 출력 ----------------
    if st.session_state.history or st.session_state.chapters:
        st.subheader("📂 면접 기록")

        # ✅ 현재 세션
        if st.session_state.history:
            st.markdown("## 🚀 현재 진행중인 면접")
            for i, res in enumerate(st.session_state.history, 1):
                render_question_result(i, res)

        # ✅ 과거 세션 (expander로 접기)
        for c_idx, chapter in enumerate(st.session_state.chapters, 1):
            with st.expander(f"📌 과거 면접 세션 {c_idx}", expanded=False):
                for i, res in enumerate(chapter, 1):
                    render_question_result(i, res)


    # ----------------------
    # 푸터
    # ----------------------

    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 14px;'>
            © 2025 AI Interview Project | Powered by Whisper · LangChain · Hailo
        </div>
        """,
        unsafe_allow_html=True
    )