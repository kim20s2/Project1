from __future__ import annotations
import streamlit as st
import requests, tempfile,time
import os, wave,uuid
import numpy as np, math
import statistics as stats, json
from pathlib import Path
from typing import Callable, Iterable
from core.analysis_audio import analyze_stability, get_stability_score
from app.adapters.posture_adapters import parse_posture_auto
from app.adapters.interviewer_adapters import my_stt_from_path as stt_fn, load_persona_videos, shuffle_order
from core.recording_io import save_assets_after_stop, BASE_DIR 
from core.chains import get_prompt,call_llm


SHOW_PER_ANSWER_METRICS = False  # 답변별 지표는 숨김
SHOW_FINAL_METRICS      = True   # 총평에서만 지표 표시

def _find_xml_for_session(session_id: str, prefer_stem: str | None = None) -> str | None:
    """
    세션 루트와 xml/ 하위 폴더를 모두 검색해서
    최신 XML을 찾는다. prefer_stem이 있으면 그걸 우선 매칭.
    (디렉터리를 새로 만들지 않도록 get_save_dir는 사용하지 않음)
    """
    session_root = BASE_DIR / session_id
    search_roots = [session_root, session_root / "xml"]

    cands = []
    for d in search_roots:
        if not d.exists():
            continue
        if prefer_stem:
            cands.extend(d.glob(f"*{prefer_stem}*.xml"))
        else:
            cands.extend(d.glob("*.xml"))

    if not cands:
        return None

    # 최신 수정시간 기준
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    return str(newest)

# 넘파이 스칼라 유틸 
def _to_native(x):
    if isinstance(x, np.generic):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v
    if isinstance(x, dict):  return {k:_to_native(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_to_native(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    return x

# path 방식 
def is_valid_wav(path: str) -> tuple[bool, str]:
    try:
        if not os.path.exists(path): return False, "파일 없음"
        if os.path.getsize(path) < 1024: return False, "파일 너무 작음"
        with wave.open(path, "rb") as w:
            if w.getnframes() <= 0: return False, "프레임 0"
        return True, "OK"
    except wave.Error as e:
        return False, f"wave 에러: {e}"
    except Exception as e:
        return False, f"기타 에러: {e}"
    
def download_wav_direct(server_url: str, max_wait_s=20, interval_s=0.5, min_bytes=16_000) -> Path:
    """
    서버에서 WAV를 단일 GET으로 스트리밍 저장하고 즉시 무결성 검증.
    - 성공 시: 총 1회 GET
    - 실패 시: 임시파일 삭제 후 재시도 (deadline까지)
    """
    url = f"{server_url}/download/wav/audio.wav"
    deadline = time.time() + max_wait_s
    last_err = None

    while time.time() < deadline:
        try:
            # 1) 단일 GET (스트리밍)
            with requests.get(url, stream=True, timeout=10) as r:
                if r.status_code != 200:
                    time.sleep(interval_s)
                    continue

                ctype = (r.headers.get("Content-Type", "") or "").lower()
                # 일부 서버는 application/octet-stream 으로 내려줄 수 있음
                if ("audio" not in ctype) and ("octet-stream" not in ctype):
                    time.sleep(interval_s)
                    continue

                # 2) 바로 파일로 저장 (청크를 버리지 않도록 곧바로 기록)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                    tmp_path = Path(tf.name)
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            tf.write(chunk)

            # 3) 빠른 크기 하한 검사 (샘플레이트/비트뎁스에 맞춰 조정 가능)
            if tmp_path.stat().st_size < min_bytes:
                try: tmp_path.unlink()
                except: pass
                time.sleep(interval_s)
                continue

            # 4) WAV 헤더/프레임 검증
            ok, reason = is_valid_wav(str(tmp_path))
            if ok:
                return tmp_path

            # 무효면 삭제 후 재시도
            try: tmp_path.unlink()
            except: pass

        except Exception as e:
            last_err = e

        time.sleep(interval_s)

    raise RuntimeError(f"WAV 준비 실패: {last_err or 'timeout'}")


def resolve_posture_xml_for(wav_path: str) -> str | None:
    """
    우선순위:
    1) WAV와 같은 폴더의 <stem>.xml
    2) .../<session>/xml/<stem>.xml
    3) .../<session>/<stem>.xml              ← 레거시 루트
    4) .../<session>/pose/<stem>.xml         ← 레거시 변형
    5) .../<session>/posture/<stem>.xml      ← 레거시 변형
    6) (최후) 세션 루트 이하 shallow glob
    """
    p = Path(wav_path)
    stem = p.stem

    # 세션 루트 추정: .../<session>/wav/<file>.wav → <session>
    session_root = p.parent.parent if p.parent.name == "wav" else p.parent

    candidates = [
        p.with_suffix(".xml"),
        session_root / "xml" / f"{stem}.xml",
        session_root / f"{stem}.xml",                # ← 추가: 루트 직하
        session_root / "pose" / f"{stem}.xml",       # ← 추가
        session_root / "posture" / f"{stem}.xml",    # ← 추가
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # 최후 수단: 동일 stem을 세션 루트 하위에서 탐색
    try:
        hits = list(session_root.glob(f"**/{stem}.xml"))
        if hits:
            return str(hits[0])
    except Exception:
        pass

    return None
def render_interviewer_panel(
    server_url: str,
    stt_fn: Callable[[str], str],
    feedback_fn: Callable[[str, str], str],
    questions: Iterable[str] = (
        "자기소개 부탁드립니다.",
        "가장 도전적이었던 프로젝트와 역할은?",
        "문제 해결 경험을 STAR 구조로 설명해 주세요.",
    ),
) -> None:
    """
    면접관 모드 패널 (MP4 전용 재생으로 수정)
    - server_url, stt_fn, feedback_fn은 아래쪽(네가 유지하는 구간)에서 그대로 사용
    """
    ss = st.session_state
    # 세션키는 eva_* 접두사로 충돌 방지
    if "eva_init" not in ss:
        ss.eva_init = True
        ss.eva_recording = False
        ss.eva_stopped = False
        ss.eva_auto_saved_once = False
        ss.eva_last_wav = None
        ss.eva_qidx = 0
        ss.eva_questions = list(questions)  # ← 하위 호환용(지금은 MP4 텍스트 사용)
        ss.eva_last_stt = ""
        ss.eva_last_fb = ""
        ss.setdefault("eva_pending_analysis", False)  # ★ 다음 런에서 분석 실행 여부
        ss.setdefault("eva_history", [])
        ss.setdefault("eva_pending_final", False)
        # MP4 재생 상태 기본값
        ss.setdefault("eva_playing", False)
        ss.setdefault("eva_answer_enabled", False)
        ss.setdefault("eva_current_idx", None)
        ss.setdefault("eva_ends_at", None)
    def _new_session():
        ss.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        ss.eva_history = []
        ss.eva_voice_summary = {}
        ss.eva_posture_summary = {}
        ss.eva_last_wav = None
        ss.eva_last_xml = None
        ss.eva_pending_analysis = False
        ss.eva_pending_final = False
        ss["_reran_to_final_once"] = False
        ss.eva_qidx = 0

        # ★ 추가 권장: 이전 세션 흔적 제거
        ss.eva_recording = False
        ss.eva_stopped = False
        ss.eva_auto_saved_once = False
        ss.eva_last_stt = ""
        ss.eva_last_fb = ""

    # ── 현재 질문 준비(영상 로드 + 랜덤 순서) ─────────────────────────────
    ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")  # app/assets
    if "eva_videos" not in ss:
        ss.eva_videos = load_persona_videos(persona=".", asset_root=ASSET_DIR)

    if "eva_order" not in ss or ss.get("eva_shuffle", False) or len(ss.eva_order) != len(ss.eva_videos):
        ss.eva_order = shuffle_order(len(ss.eva_videos))   # 무중복 랜덤
        ss.eva_qidx = 0
        ss.eva_shuffle = False

    cur_idx = ss.eva_order[ss.eva_qidx]
    cur = ss.eva_videos[cur_idx]
    ss.setdefault("eva_started", False)
    ss.setdefault("eva_playing", False)
    ss.setdefault("eva_answer_enabled", False)
    ss.setdefault("eva_current_idx", None)
    ss.setdefault("eva_ends_at", None)
    ss.setdefault("eva_cur_dur", None)

    MAX_WAIT = 20.0  # UX 상한

    def _start_play(idx: int, q: dict):
        ss.eva_started = True
        ss.eva_playing = True
        ss.eva_answer_enabled = False
        ss.eva_current_idx = idx
        ss.eva_cur_dur = min(q["duration"], MAX_WAIT)
        ss.eva_ends_at = time.time() + ss.eva_cur_dur
        st.rerun()
    c1, c2, c3 = st.columns([1, 1, 1])

  # ── c1: 면접 시작 버튼
    with c1:
        if st.button("▶ 면접 시작", use_container_width=True,
                    disabled=ss.get("eva_started", False) or ss.get("eva_recording", False)):
            _new_session()            
            _start_play(cur_idx, cur)
    # ── 면접관 영상은 mid에서 렌더
    left, mid, right = st.columns([2.5, 2, 2.5])
    with mid:
        if ss.get("eva_playing") and ss.get("eva_current_idx") is not None:
          st.video(ss.eva_videos[ss.eva_current_idx]["mp4"], start_time=0)

    # ── 타이머: 조용히 갱신해서 끝나면 자동 활성화
    if ss.get("eva_ends_at"):
        remain = ss.eva_ends_at - time.time()
        if remain <= 0:
            ss.eva_playing = False
            ss.eva_answer_enabled = True
            ss.eva_ends_at = None
        else:
            time.sleep(min(0.5, max(0.1, remain)))
            st.rerun()

        # ── c3: '다음 질문'은 시작 이후엔 항상 보이지만, 조건에 따라 disabled
    with right:
        if ss.get("eva_started", False):
            next_disabled = (
                ss.get("eva_playing", False)               # 재생 중이면 비활성
                or not ss.get("eva_answer_enabled", False) # 답변 가능 상태 아니면 비활성
                or ss.get("eva_recording", False)          # 녹음 중이면 비활성
            )
            
            clicked = st.button("➡ 다음 질문", key="btn_next", use_container_width=True, disabled=next_disabled)

            if clicked:
                order = ss.eva_order
                if not order:
                    st.warning("다음 질문 목록이 비어 있어요.")
                else:
                    # 인덱스는 order 길이에 맞춰 회전
                    ss.eva_qidx = (ss.eva_qidx + 1) % len(order)
                    next_idx = order[ss.eva_qidx]
                    next_q   = ss.eva_videos[next_idx]
                    _start_play(next_idx, next_q)  # 누르는 즉시 다음 영상 자동 재생

    with c2:
        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("▶️ 답변 시작", use_container_width=True, disabled=ss.eva_recording, key="btn_start"):
                COUNTDOWN = 5
                msg = st.empty()
                bar = st.progress(0, text="곧 면접을 시작합니다…")
                try:
                    # 1) 서버 녹음 먼저 시작
                    r = requests.get(f"{server_url}/command/start_record", timeout=(3.05, 8))
                    r.raise_for_status()

                    # 2) 상태 갱신(버튼 비활성화 등)
                    ss.eva_recording = True
                    ss.eva_stopped = False
                    ss.eva_auto_saved_once = False
                    ss.eva_last_wav = None

                    # 3) 그 다음 카운트다운(녹음은 이미 진행 중)
                    for sec in range(COUNTDOWN, 0, -1):
                        msg.info(f"🕒 {sec}초 후 답변 시작")
                        bar.progress(int(((COUNTDOWN - sec + 1) / COUNTDOWN) * 100))
                        time.sleep(1)

                    st.success("면접이 시작되었습니다.\n답변을 말씀해 주세요.")
                except requests.exceptions.RequestException as e:
                    st.error(f"요청 실패: {e}")
                finally:
                    bar.empty()
                    msg.empty()

        with col_stop:
            if st.button("⏹️ 답변 종료", use_container_width=True, disabled=not ss.eva_recording, key="btn_stop"):
                try:
                    # 1) 녹음 종료
                    r = requests.get(f"{server_url}/command/stop_record", timeout=60)
                    r.raise_for_status()

                    # ★ 상태 업데이트만 하고, 긴 작업은 다음 런에서 수행
                    ss.eva_recording = False
                    ss.eva_stopped = True
                    ss.eva_pending_analysis = True   # 다음 런에서 run_analysis_if_needed()가 실행
                    ss.eva_auto_saved_once = False
                    # ★ 스냅샷(이 답변은 어떤 세션/몇 번째 질문이었는지 고정)
                    ss.eva_session_for_answer = ss.session_id
                    ss.eva_qidx_for_answer    = ss.eva_qidx
                    st.success("답변이 종료되었습니다.\n분석을 시작합니다…")

                    # ★ 즉시 UI 재렌더 → '답변 시작' 버튼이 바로 활성화됨
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"요청 실패: {e}")
    # 3) 면접 종료(세션 종료)
    with c3:
        end_enabled = (
        (not ss.get("eva_recording", False))
        and (len(ss.get("eva_history", [])) > 0 or ss.get("eva_pending_analysis", False))
        and (not ss.get("eva_pending_final", False))  # ← 추가!
    )
        if st.button("🏁 면접 종료", use_container_width=True, disabled=not end_enabled, key="btn_end"):
            ss.eva_pending_final = True    # ✅ 총평을 실행하라는 신호만 남기기
            st.rerun()                     # ✅ 다음 런에서 총평 블록 실행
    # 면접 내용 저장(mp4, wav, xml)
    if ss.eva_stopped and not ss.eva_auto_saved_once:
        kinds = ["wav", "xml"] if ss.get("mp4_manual_only") else ["mp4", "wav", "xml"]
            # ★ 스냅샷으로 고정
        sess_for_save = ss.get("eva_session_for_answer", ss.session_id)
        qidx_for_save = ss.get("eva_qidx_for_answer", ss.eva_qidx)

        with st.spinner("저장 중…"):
            saved = save_assets_after_stop(
            server_url=server_url,
            session_id=sess_for_save,   # ← 현재 세션이 아니라 정지 당시 세션
            kinds=tuple(kinds),          # ← iterable이면 리스트도 OK, 습관상 튜플로
            qidx=qidx_for_save,          # 정지 당시 질문 인덱스 
            # stem=원하면_외부에서_고정_stem_지정_가능
        )
        if saved:
            # ★ 저장된 로컬 경로를 세션에 고정해 둔다
            if "wav" in saved: ss.eva_last_wav  = str(saved["wav"])
            if "xml" in saved: ss.eva_last_xml = str(saved["xml"])
            ss.eva_auto_saved_once = True
            st.success("저장 완료!")

    answer_box = st.container()  # 방금 답변 결과(텍스트/피드백만)
    final_box  = st.container()  # 면접 종료 총평 표시
    if ss.eva_pending_analysis:
        with answer_box:
            try:
                # 1) WAV 경로 확보 (없으면 다운로드)
                if not ss.eva_last_wav:
                    wav_path = download_wav_direct(server_url, max_wait_s=30, interval_s=0.5)
                    ss.eva_last_wav = str(wav_path)

                # 2) (보너스) 파일이 없어졌으면 다시 다운로드
                if not os.path.exists(ss.eva_last_wav):
                    wav_path = download_wav_direct(server_url, max_wait_s=30, interval_s=0.5)
                    ss.eva_last_wav = str(wav_path)

                # 3) 유효성 검사 → STT(경로 기반)
                ok, info = is_valid_wav(ss.eva_last_wav)
                if not ok:
                    st.error(f"오디오 파일 이상: {info}")
                    ss.eva_pending_analysis = False
                    st.stop()

                st.info("답변 텍스트로 변환중…")
                ss.eva_last_stt = stt_fn(ss.eva_last_wav)  # my_stt_from_path

                # 현재 질문 텍스트
                qidx_for = ss.get("eva_qidx_for_answer", ss.eva_qidx)
                order = ss.get("eva_order") or []
                if not order or not isinstance(qidx_for, int) or qidx_for >= len(order):
                    qtext = f"Q{(qidx_for if isinstance(qidx_for, int) else 0)+1}"
                else:
                    cur_idx = order[qidx_for]
                    cur = ss.eva_videos[cur_idx]
                    qtext = cur.get("text") or cur.get("caption") or f"Q{qidx_for+1}"

                # 4) 음성 피처/안정성
                pitch_metrics = {}
                try:
                    feats, _ = analyze_stability(ss.eva_last_wav)
                    score, label, _ = get_stability_score(
                        feats.get("jitter", 0.0),
                        feats.get("shimmer", 0.0),
                        feats.get("hnr", 0.0),
                    )
                    pitch_metrics = {**feats, "stability_score": score, "stability_label": label}
                except Exception:
                    pass

                # 5) XML 경로(세이프가드 포함) 확보 → 자세 파싱(어댑터)
                xml_path = ss.get("eva_last_xml") or resolve_posture_xml_for(ss.eva_last_wav)
                if not xml_path:
                    from pathlib import Path
                    # _find_xml_for_session()이 있다면 폴백 사용(없으면 이 블록 삭제해도 됨)
                    try:
                        xml_path = _find_xml_for_session(
                            ss.get("session_id", ""),
                            prefer_stem=(Path(ss.eva_last_wav).stem if ss.get("eva_last_wav") else None),
                        )
                    except NameError:
                        xml_path = None  # 함수가 없으면 무시

                posture_metrics = {}
                try:
                    if xml_path:
                        flavor, posture_common, _ = parse_posture_auto(xml_path)

                        # (히스토리/집계 호환 별칭)
                        posture_for_hist = dict(posture_common)
                        posture_for_hist.setdefault("head_tilt_count", posture_for_hist.get("negative_emotion_count", 0))
                        posture_for_hist.setdefault("body_tilt_count", posture_for_hist.get("bad_posture_count", 0))
                        posture_for_hist.setdefault("gesture_count",  posture_for_hist.get("leg_shake_count", 0))

                        # 세션 보관
                        ss.eva_posture_summary = posture_for_hist
                        ss.eva_posture_label   = posture_common.get("label", "정상")

                        # 수집 범위(LLM 추정 방지)
                        ss.eva_posture_avail = (
                            {"negative_emotion": True, "bad_posture": True, "leg_shake": True}
                            if flavor == "jetson" else
                            {"negative_emotion": False, "bad_posture": True, "leg_shake": False}
                        )

                        posture_metrics = posture_for_hist
                    else:
                        ss.eva_posture_summary = {}
                        ss.eva_posture_label   = "데이터 없음"
                except Exception as e:
                    st.warning(f"자세 파싱 실패: {e}")
                    ss.eva_posture_summary = {}
                    ss.eva_posture_label   = "데이터 없음"

                # 6) 피드백(자세 수집범위 + 라벨 섞어서 전달)  ← 반드시 '자세 파싱' 이후
                a = ss.get("eva_posture_avail", {})
                scope_text = (
                    f"[자세 수집 범위] "
                    f"표정:{'있음' if a.get('negative_emotion') else '없음'}, "
                    f"자세:{'있음' if a.get('bad_posture') else '없음'}, "
                    f"다리떨림:{'있음' if a.get('leg_shake') else '없음'}"
                )
                posture_label = ss.get("eva_posture_label") or "정상"

                metrics_line = ""
                if posture_metrics:
                    frames = posture_metrics.get("frames")
                    head   = posture_metrics.get("head_tilt_count")
                    body   = posture_metrics.get("body_tilt_count")
                    gest   = posture_metrics.get("gesture_count")
                    metrics_line = f"[자세 수치] frames={frames}, head={head}, body={body}, gesture={gest}"

                aug_answer = f"{ss.eva_last_stt}\n\n{scope_text}\n[자세 요약] {posture_label}\n{metrics_line}"
                ss.eva_last_fb = feedback_fn(qtext, aug_answer)
                st.info("피드백 생성 중…")
                ss.eva_last_fb = feedback_fn(qtext, aug_answer)

                # 세션 요약(음성) 최신화만 갱신
                ss.eva_voice_summary = pitch_metrics or ss.get("eva_voice_summary") or {}

                # 화면 출력
                st.markdown("### 📝 이번 답변")
                st.write(ss.eva_last_stt or "(빈 텍스트)")
                st.markdown("### 🎯 피드백")
                st.write(ss.eva_last_fb or "(피드백 없음)")

                # 7) 히스토리 저장
                ss.eva_history.append({
                    "session_id": ss.session_id,
                    "qidx": ss.eva_qidx,
                    "qtext": qtext,
                    "wav_path": ss.eva_last_wav,
                    "stt": ss.eva_last_stt,
                    "fb": ss.eva_last_fb,
                    "pitch": pitch_metrics,
                    "posture": posture_metrics,  # ← 방금 만든 것 저장
                })
                if ss.eva_questions and ss.eva_qidx < len(ss.eva_questions) - 1:
                    ss.eva_qidx += 1

                st.success("분석 완료")

            except Exception as e:
                st.exception(e)
            finally:
                ss.eva_pending_analysis = False
                ss.eva_session_for_answer = None
                ss.eva_qidx_for_answer    = None

                    
                 # ✅ 방금 분석을 끝냈고 사용자가 미리 🏁 눌러둔 경우에만 1번 rerun
    if ss.eva_pending_final and not ss.get("_reran_to_final_once", False):
        ss["_reran_to_final_once"] = True
        st.rerun()

    elif ss.eva_pending_final and not ss.eva_pending_analysis:
        
        with final_box:
          with st.spinner("🧾 총평 생성 중…"):
            try:
                # --- (그대로 유지) 지표 집계 ---
                def _avg(seq):
                    xs = [x for x in seq if x is not None]
                    return stats.mean(xs) if xs else None

                getp = lambda k: [h.get("pitch",{}).get(k) for h in ss.eva_history]
                summary_voice = {
                    "avg_stability_score": _to_native(_avg(getp("stability_score"))),
                    "avg_jitter":          _to_native(_avg(getp("jitter"))),
                    "avg_shimmer":         _to_native(_avg(getp("shimmer"))),
                    "avg_hnr":             _to_native(_avg(getp("hnr"))),
                    "avg_f0_std":          _to_native(_avg(getp("f0_std"))),
                    "avg_loudness_std":    _to_native(_avg(getp("loudness_std"))),
                }

                # --- 자세 집계 (공통 스키마로 합산) ---
                total_frames = sum((h.get("posture",{}).get("frames") or 0) for h in ss.eva_history)
                sum_neg = sum((h.get("posture",{}).get("negative_emotion_count") or 0) for h in ss.eva_history)
                sum_bad = sum((h.get("posture",{}).get("bad_posture_count") or 0) for h in ss.eva_history)
                sum_leg = sum((h.get("posture",{}).get("leg_shake_count") or 0) for h in ss.eva_history)

                def _sev(n, d):
                    if not d or n is None: return "정상"
                    r = float(n)/float(d)
                    return "강" if r >= 0.40 else "중" if r >= 0.20 else "약" if r >= 0.05 else "정상"

                summary_posture_common = {
                    "frames": int(total_frames),
                    "negative_emotion_count": sum_neg,
                    "negative_emotion_severity": _sev(sum_neg, total_frames),
                    "bad_posture_count": sum_bad,
                    "bad_posture_severity": _sev(sum_bad, total_frames),
                    "leg_shake_count": sum_leg,
                    "leg_shake_severity": _sev(sum_leg, total_frames),
                }

                _labels = []
                if summary_posture_common["negative_emotion_severity"] != "정상":
                    _labels.append(f"표정(부정):{summary_posture_common['negative_emotion_severity']}")
                if summary_posture_common["bad_posture_severity"] != "정상":
                    _labels.append(f"자세:{summary_posture_common['bad_posture_severity']}")
                if summary_posture_common["leg_shake_severity"] != "정상":
                    _labels.append(f"다리 떨림:{summary_posture_common['leg_shake_severity']}")
                summary_posture_common["label"] = ", ".join(_labels) if _labels else "정상"

                posture_json = json.dumps(summary_posture_common, ensure_ascii=False)

                # (선택) 없으면 언급하지 않기: availability로 필터링
                def _filter_posture_for_llm(posture_common: dict, avail: dict) -> dict:
                    avail = avail or {}
                    keep = {
                        "frames": posture_common.get("frames", 0),
                        "label": posture_common.get("label", ""),
                    }
                    if avail.get("negative_emotion"):
                        keep["negative_emotion_count"]    = posture_common.get("negative_emotion_count")
                        keep["negative_emotion_severity"] = posture_common.get("negative_emotion_severity")
                    if avail.get("bad_posture"):
                        keep["bad_posture_count"]    = posture_common.get("bad_posture_count")
                        keep["bad_posture_severity"] = posture_common.get("bad_posture_severity")
                    if avail.get("leg_shake"):
                        keep["leg_shake_count"]    = posture_common.get("leg_shake_count")
                        keep["leg_shake_severity"] = posture_common.get("leg_shake_severity")
                    return keep

                posture_avail = ss.get("eva_posture_avail", {})  # 분석 단계에서 flavor로 세팅해둔 값
                posture_to_send = _filter_posture_for_llm(summary_posture_common, posture_avail)

                # --- LLM에 문자열로 주입 (필요한 것만) ---
                voice_json   = json.dumps(summary_voice,   ensure_ascii=False)
                posture_json = json.dumps(posture_to_send, ensure_ascii=False)

                history_compact = [{
                    "q": h["qtext"],
                    "stt": (h.get("stt","")[:150] + "…") if h.get("stt") and len(h["stt"])>150 else (h.get("stt") or "")
                } for h in ss.eva_history[-8:]]
                history_compact_json = json.dumps(history_compact, ensure_ascii=False)
                avail_json = json.dumps(ss.get("eva_posture_avail", {}), ensure_ascii=False)
                messages = get_prompt("session").format_messages(
                        voice_summary_json=voice_json,
                        posture_summary_json=posture_json,
                        posture_availability_json=avail_json,  # ← 복원
                        history_compact_json=history_compact_json,
                        answers=len(ss.get("eva_history", [])),
)

                summary_text = call_llm(messages)
                st.markdown("## 🧾 면접 결과")
                st.write(summary_text)
                st.success("면접 총평 생성 완료")

                # 다음 렌더에서도 보여줄 수 있게
                ss.eva_last_summary = summary_text
                ss.eva_show_last_summary = True

                # --- finally: 세션 아카이브 + 리셋 (들여쓰기 주의) ---
            finally:
                    voice_dict   = summary_voice
                    posture_dict = summary_posture_common  # (필요하면 posture_to_send를 저장해도 됨)
                    ss.setdefault("eva_sessions", []).append({
                        "session_id": ss.session_id,
                        "answers": ss.eva_history.copy(),
                        "summary": summary_text if 'summary_text' in locals() else None,
                        "voice":   voice_dict,
                        "posture": posture_dict,
                        "ended_at": time.time(),
                    })
                    ss.eva_history = []  # 다음 면접을 위해 비움

                    # ★ 재시작을 막는 상태값들 리셋
                    ss.eva_started = False
                    ss.eva_playing = False
                    ss.eva_answer_enabled = False
                    ss.eva_current_idx = None
                    ss.eva_ends_at = None
                    ss.eva_recording = False
                    ss.eva_stopped = False
                    ss.eva_auto_saved_once = False

                    # 플래그 정리
                    ss.eva_pending_final = False
                    ss["_reran_to_final_once"] = False

                    # 버튼 상태 최신화 위해 한 번 더 렌더
                    st.rerun()
        # 총평을 다음 렌더에서 다시 보여주기 (rerun 후에도 보이게)
    if ss.get("eva_show_last_summary"):
        with final_box:
            st.markdown("## 🧾 면접 결과")
            st.write(ss.get("eva_last_summary") or "(요약 없음)")
            st.success("면접 총평 생성 완료")
        # 한 번 보여줬으면 플래그 내림
        ss.eva_show_last_summary = False
    # 상태 표시
    st.markdown("🟢 **답변 중입니다...**" if ss.eva_recording else "⚪ **면접 대기 중**")
    st.markdown("---")
