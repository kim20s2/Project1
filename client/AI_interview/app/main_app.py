import os, time, tempfile
from pathlib import Path
import requests
import streamlit as st
from app.upload_section import render_upload_section
from core.storage import get_save_dir, safe_name, save_bytes
from core.remote_fetch import wait_until_ready
from core.analysis_pose import parse_posture_summary
from core.whisper_run import load_whisper, transcribe_file
from core.chains import build_feedback_chain
from core.analysis_audio import analyze_stability, get_stability_score
from interviewer import render_interviewer_panel
from adapters.interviewer_adapters import my_tts_interviewer, my_stt_from_path, my_feedback
# ── 페이지/세션 기본 설정 ─────────────────────────────────────────────
st.set_page_config(page_title="AI 면접관", page_icon="🎤", layout="wide")

ss = st.session_state
ss.setdefault("stopped", False)
ss.setdefault("recording", False)
ss.setdefault("auto_mp4", True)
ss.setdefault("mp4_manual_only", False)     # MP4 수동 받기 
ss.setdefault("auto_saved_once", False)
ss.setdefault("session_id", "default_session")

# ── 외부 서버 주소(라즈베리파이 Flask) ───────────────────────────────
SERVER_URL = os.environ.get("PI_SERVER_URL", "http://10.10.14.80:5000")

# ── 저장 경로 준비(업로드 모드) ────────────────────────────────────────────────────
BASE_DIR = Path("downloaded")
SAVE_DIR = get_save_dir(BASE_DIR, ss["session_id"])

# ── 모델/체인(세션당 1회 로드) ────────────────────────────────────────
if "whisper_model" not in ss:
    ss["whisper_model"] = load_whisper("small")  # small/medium/large

def load_google_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return ""

GOOGLE_API_KEY = load_google_key()
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. 환경변수 또는 .streamlit/secrets.toml로 설정해주세요.")
    st.stop()

# --- 체인: 세션당 1회만 생성 ---
ss = st.session_state
if "feedback_chain" not in ss:
    ss["feedback_chain"] = build_feedback_chain(
        api_key=GOOGLE_API_KEY,
        model="gemini-2.5-flash",
        temperature=0.6,
    )

# ── UI 타이틀 ─────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF; font-size: 40px;'>🎤 AI 면접관</h1>
    <h3 style='text-align: center; color: #555;'>당신의 <b>답변 · 자세 · 표정</b>을 종합 분석합니다</h3>
    """,
    unsafe_allow_html=True
)
st.divider()

# 실시간 면접 세션 
render_interviewer_panel(
    server_url=SERVER_URL,
    tts_interviewer=my_tts_interviewer,
    stt_fn=my_stt_from_path,
    feedback_fn=my_feedback,
)

# 업로드 세션 
render_upload_section(
    SAVE_DIR=SAVE_DIR,          # Path 객체 (예: downloaded/<SESSION_ID>)
    ss=st.session_state,        # 세션 상태
    whisper_model=ss["whisper_model"],
    feedback_chain=ss["feedback_chain"],
)

st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>© 2025 AI Interview Project | Powered by Streamlit · Whisper · LangChain · Hailo</div>",
    unsafe_allow_html=True
)
