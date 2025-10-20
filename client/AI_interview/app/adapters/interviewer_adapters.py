from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import os, io, wave, json, glob, subprocess
from pathlib import Path
from functools import lru_cache
from core import whisper_run, chains  # <- 방금 이름 바꾼 파일을 임포트

# 재노출 목록(기존 __all__ 있으면 병합)
try:
    __all__  # type: ignore
except NameError:
    __all__ = []  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# 1) TTS 더미: 항상 유효한 WAV(무음) 반환  → TTS 미사용시에도 파이프라인 안정
DEFAULT_SR = int(os.getenv("TTS_SAMPLE_RATE_HZ", "16000"))

def _wav_silence(seconds: float = 0.1, sr: int = DEFAULT_SR) -> bytes:
    """seconds 길이의 16kHz mono 16-bit PCM 무음 WAV 바이트 생성(플레이어/인코더가 항상 인식)."""
    n = max(1, int(sr * float(seconds)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)      # 16-bit
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()

def my_tts_interviewer(text: str, speed: float = 0.95) -> Tuple[bytes, str]:
    """
    현재 모드: TTS 비사용. 항상 유효한 무음 WAV 반환.
    - 반환: (wav_bytes, "audio/wav")
    - 추후 TTS 붙일 때 여기서 backend 분기 추가하면 됨.
    """
    return (_wav_silence(0.1, DEFAULT_SR), "audio/wav")

__all__ += ["my_tts_interviewer"]

# ──────────────────────────────────────────────────────────────────────────────
# 2) 비디오 로딩 유틸: assets/interviewer/<persona>/ 에서 질문 목록 구성

def _default_asset_root() -> str:
    # 이 파일 기준 ../assets/interviewer
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "assets", "interviewer")

def _probe_duration(path: str, fallback: float = 8.0) -> float:
    """ffprobe로 길이(초) 추정. ffprobe 실패 시 fallback(초)."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], timeout=5).decode().strip()
        return max(0.0, float(out))
    except Exception:
        return float(fallback)

def _text_from_filename(basename: str) -> str:
    """
    파일명에서 질문 텍스트 추출:
    - q001_자기소개.mp4 → '자기소개'
    - q002.mp4 → '질문 q002'
    """
    stem = os.path.splitext(basename)[0]
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].replace("-", " ").strip()
    return f"질문 {parts[0]}"

def load_persona_videos(
    persona: str,
    asset_root: Optional[str] = None,
    *,
    fallback_duration: float = 8.0
) -> List[Dict]:
    """
    assets/interviewer/<persona>/ 에서 질문 비디오 목록을 로드.
    우선순위: manifest.json → q*.mp4 스캔
    반환 각 항목: { "id": str, "text": str, "mp4": str(절대경로), "duration": float }
    """
    root = asset_root or os.getenv("ASSET_ROOT") or _default_asset_root()
    base = os.path.join(root, persona)
    if not os.path.isdir(base):
        raise RuntimeError(f"면접관 폴더가 없습니다: {base}")

    # 1) manifest.json 우선
    man_path = os.path.join(base, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        out: List[Dict] = []
        for q in man.get("questions", []):
            mp4_path = os.path.join(base, q["mp4"])
            if not os.path.exists(mp4_path):
                continue
            dur = float(q.get("duration_s") or _probe_duration(mp4_path, fallback=fallback_duration))
            out.append({
                "id": q["id"],
                "text": q.get("text") or f"질문 {q['id']}",
                "mp4": mp4_path,
                "duration": dur,
            })
        if out:
            __all__.append("load_persona_videos")
            return out  # manifest 존재 시 여기서 종료

    # 2) manifest 없거나 비어 있으면 q*.mp4 스캔
    mp4s = sorted(glob.glob(os.path.join(base, "q*.mp4")))  # q001*.mp4 허용
    if not mp4s:
        raise RuntimeError(f"mp4를 찾지 못했습니다: {base}")

    items: List[Dict] = []
    for p in mp4s:
        bn = os.path.basename(p)
        qid = os.path.splitext(bn)[0].split("_")[0]
        items.append({
            "id": qid,
            "text": _text_from_filename(bn),
            "mp4": p,
            "duration": _probe_duration(p, fallback=fallback_duration),
        })
    __all__.append("load_persona_videos")
    return items

# ──────────────────────────────────────────────────────────────────────────────
# 3) 랜덤 순서 유틸

def shuffle_order(n: int, seed: Optional[int] = None) -> List[int]:
    """0..n-1 인덱스를 무작위로 섞어서 반환(무중복). seed로 재현 가능."""
    import random
    order = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(order)
    return order

__all__ += ["shuffle_order"]

@lru_cache(maxsize=1)
def _get_whisper_model():
    # 필요 시 사이즈 변경: "base", "small", "medium", "large"
    return whisper_run.load_whisper(model_size="small")

def my_stt_from_path(wav_path: str) -> str:
    """
    - 파일 경로를 그대로 Whisper에 전달해 STT 수행
    - 반환: result["text"]
    """
    model = _get_whisper_model()  # 네 파일에 이미 있는 헬퍼 그대로 사용
    result = whisper_run.transcribe_file(
        model,
        wav_path,               # ← 경로 직접 전달
        language="ko",
        task="transcribe",
        temperature=0.0,
    )
    return (result.get("text") or "").strip()

def my_feedback(question: str, answer: str, nonverbal: str | None = None) -> str:
    if not (answer and answer.strip()):
        return "답변이 비어 있어요. 한 문장으로 핵심 결론부터 말해보세요."
    latest_utt = (answer or "")[:800]                 # 방어적 슬라이스
    nv = (nonverbal or "특이사항 없음")[:300]
    try:
        chain = chains.get_live_feedback_chain()        # 캐시된 LLMChain
        return chain.run({"latest_utt": latest_utt, "nonverbal": nv})
    except Exception as e:
        return f"(임시 코칭) 첫 문장에 결론을 제시하고, 사례/수치를 1개만 보태 보세요. // 오류: {e}"
 