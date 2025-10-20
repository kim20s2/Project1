from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict,Iterable
import requests, time, re,uuid

BASE_DIR = Path("data/records")
KIND_EXTS = {"wav": ".wav", "xml": ".xml", "mp4": ".mp4"}  # 참고용

def get_save_dir(session_id: str, kind: str | None = None) -> Path:
    """
    세션 루트 또는 형식별 하위 폴더 경로를 반환.
    - kind=None  -> data/records/<session_id>/
    - kind="wav" -> data/records/<session_id>/wav/
      kind="xml" -> data/records/<session_id>/xml/
      kind="mp4" -> data/records/<session_id>/mp4/
    """
    d = BASE_DIR / session_id
    if kind:
        d = d / kind
    d.mkdir(parents=True, exist_ok=True)
    return d

def wait_until_ready(url: str, max_wait_s: int = 10, interval_s: float = 0.5) -> Optional[requests.Response]:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=(3.05, 8))
            if r.ok and r.content:
                return r
        except Exception:
            pass
        time.sleep(interval_s)
    return None

# 파일 이름 정규화 함수 (이름 정리 + 세션표시 + 확장자 유지)
def safe_name(stem: str, session_id: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    sid  = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id or "sess")
    return f"{stem}_{sid}{suffix}"

def save_bytes(dirpath: Path, filename: str, content: bytes) -> Path:
    p = dirpath / filename
    p.write_bytes(content)
    return p

# 고유 stem 생성 함수 
def make_unique_stem(qidx: int | None = None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:4]
    return f"{ts}_q{qidx:02d}_{short}" if isinstance(qidx, int) else f"{ts}_{short}"

def save_assets_after_stop(
    server_url: str,
    session_id: str,
    kinds: Iterable[str] = ("mp4", "wav", "xml"),  # ✅ tuple/iterable default
    *,
    qidx: Optional[int] = None,        # ✅ Streamlit 없이 주입
    stem: Optional[str] = None,        # (선택) 외부에서 고정 stem을 직접 지정 가능
) -> Dict[str, Path]:
    """
    서버의 고정 경로에서 mp4/wav/xml을 받아 세션/<kind>/ 하위 폴더에 저장.
    세 포맷 모두 같은 stem을 사용해 덮어쓰기 방지 + 매칭 용이.
    예: data/records/<session_id>/{wav,xml,mp4}/<stem>_<session_id>.<ext>
    """
    files = {
        "mp4": {"url": f"{server_url}/download/mp4/video.mp4", "name": "video.mp4"},
        "wav": {"url": f"{server_url}/download/wav/audio.wav", "name": "audio.wav"},
        "xml": {"url": f"{server_url}/download/xml/log.xml",   "name": "log.xml"},
        # "mp4": {"url": f"{server_url}/download/mp4/video_ai.mp4", "name": "video_ai.mp4"},
    }

    out: Dict[str, Path] = {}

    # ★ 한 번 생성한 고유 stem을 모든 포맷에 재사용
    unique_stem = stem or make_unique_stem(qidx)   # 예) 20251019_2013_q03_ab12

    for k in kinds:
        info = files.get(k)
        if not info:
            continue

        res = wait_until_ready(info["url"], max_wait_s=10, interval_s=0.5)
        if not (res and res.content):
            continue

        dest_dir = get_save_dir(session_id, k)       # .../<session>/<k>/
        suffix   = Path(info["name"]).suffix         # ".wav"/".xml"/".mp4"
        filename = safe_name(unique_stem, session_id, suffix)  # 같은 stem 재사용
        out[k]   = save_bytes(dest_dir, filename, res.content)

    return out