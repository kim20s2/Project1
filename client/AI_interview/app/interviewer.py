from __future__ import annotations
import streamlit as st
import requests, tempfile, base64, shutil,time
import os, wave,glob,uuid
import numpy as np, math
import time 
import statistics as stats, json
from pathlib import Path
from typing import Callable, Iterable, Tuple
from core.analysis_audio import analyze_stability, get_stability_score
from core.analysis_pose import parse_posture_summary, normalize_posture
from adapters.interviewer_adapters import my_stt_from_path as stt_fn, load_persona_videos, shuffle_order
from core.recording_io import save_assets_after_stop, BASE_DIR 
from core.chains import get_prompt,call_llm
from adapters.posture_adapters import parse_posture_auto, build_nonverbal_json

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

#path 방식 
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
    1) WAV와 같은 폴더의 <stem>.xml (구조 변경 전 호환)
    2) .../<session>/xml/<stem>.xml (형식별 폴더 분리 후)
    """
    p = Path(wav_path)

    # 1) 같은 폴더
    same = p.with_suffix(".xml")
    if same.exists():
        return str(same)

    # 2) 세션 루트/xml/<stem>.xml
    #    보통 wav 경로: .../<session>/wav/<stem>.wav
    #    예외로 루트에 저장된 경우도 고려
    try:
        if p.parent.name == "wav":
            session_root = p.parent.parent
        else:
            # wav가 세션 루트 바로 아래 있는 구버전 구조
            session_root = p.parent

        cand = session_root / "xml" / (p.stem + ".xml")
        if cand.exists():
            return str(cand)
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

  # ── c1: 면접 시작(처음 1회만 활성화)
    with c1:
        if st.button("▶ 면접 시작", use_container_width=True,
                    disabled=ss.get("eva_started", False) or ss.get("eva_recording", False)):
            _new_session()            # ← 여기!
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

                    st.success("면접이 시작되었습니다. 답변을 말씀해 주세요.")
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
                    st.success("답변이 종료되었습니다. 분석을 시작합니다…")

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
            session_id=ss.session_id,   # ← get(...) 대신 확실한 세션 ID 사용 권장
            kinds=tuple(kinds),          # ← iterable이면 리스트도 OK, 습관상 튜플로
            qidx=ss.eva_qidx,           # ★ 추가: 녹음 당시 질문 인덱스 전달
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
                    # 1) 경로 확보 (없으면 다운로드)
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
                    ss.eva_last_stt = stt_fn(ss.eva_last_wav)   # my_stt_from_path 사용

                    qtext = ss.eva_questions[ss.eva_qidx] if ss.eva_questions else f"Q{ss.eva_qidx+1}"
                    st.info("피드백 생성 중…")
                    ss.eva_last_fb = feedback_fn(qtext, ss.eva_last_stt)

                    # (선택) 음정/자세 계산해서 저장만
                    pitch_metrics, posture_metrics = {}, {}
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
                    try:
                        xml_path = ss.get("eva_last_xml")
                        if not xml_path:
                            xml_path = resolve_posture_xml_for(ss.eva_last_wav)  # wav와 같은 이름의 .xml
                        if not xml_path:
                            stem = Path(ss.eva_last_wav).stem if ss.get("eva_last_wav") else None
                            xml_path = _find_xml_for_session(ss.get("session_id", "sess"), prefer_stem=stem)

                        if xml_path and os.path.exists(xml_path):
                            raw = parse_posture_summary(xml_path) or {}
                            posture_metrics = normalize_posture(raw)  # ★ 정규화해서 키 통일
                        else:
                            st.warning(f"자세 XML을 찾지 못함: wav={ss.get('eva_last_wav')}, xml={xml_path}")
                    except Exception as e:
                        st.warning(f"자세 요약 파싱 실패: {e}")
                    ss.eva_voice_summary   = pitch_metrics or ss.get("eva_voice_summary")   or {}
                    ss.eva_posture_summary = posture_metrics or ss.get("eva_posture_summary") or {}
                    # 화면 출력
                    st.markdown("### 📝 이번 답변")
                    st.write(ss.eva_last_stt or "(빈 텍스트)")
                    st.markdown("### 🎯 피드백")
                    st.write(ss.eva_last_fb or "(피드백 없음)")

                    # ✅ 히스토리 저장
                    ss.eva_history.append({
                        "session_id": ss.session_id,
                        "qidx": ss.eva_qidx,
                        "qtext": qtext,
                        "wav_path": ss.eva_last_wav,
                        "stt": ss.eva_last_stt,
                        "fb": ss.eva_last_fb,
                        "pitch": pitch_metrics,
                        "posture": posture_metrics,
                    })
                    if ss.eva_questions and ss.eva_qidx < len(ss.eva_questions) - 1:
                        ss.eva_qidx += 1

                    st.success("분석 완료")
                except Exception as e:
                    st.exception(e)
                finally:
                    ss.eva_pending_analysis = False  # ✅ 여기서 확실히 내리기
                    ss.eva_session_for_answer = None   # 스냅샷 해제
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

                total_frames = sum((h.get("posture",{}).get("frames") or 0) for h in ss.eva_history)
                sum_head    = sum((h.get("posture",{}).get("head_tilt_count") or 0) for h in ss.eva_history)
                sum_body    = sum((h.get("posture",{}).get("body_tilt_count") or 0) for h in ss.eva_history)
                sum_gesture = sum((h.get("posture",{}).get("gesture_count") or 0) for h in ss.eva_history)
                def _rate(n, d): return (n / d) if d and n is not None else None
                summary_posture = {
                    "head_tilt_rate":    _to_native(_rate(sum_head, total_frames)),
                    "body_tilt_rate":    _to_native(_rate(sum_body, total_frames)),
                    "gesture_per_frame": _to_native(_rate(sum_gesture, total_frames)),
                    "frames_total":      int(total_frames),
                }

                posture_dict = summary_posture or ss.get("eva_posture_summary") or {}
                voice_dict   = summary_voice   or ss.get("eva_voice_summary")   or {}

                posture_json = json.dumps(posture_dict, ensure_ascii=False)  # ← 끝에 쉼표(,) 금지!
                voice_json   = json.dumps(voice_dict,   ensure_ascii=False)

                # --- (화면 출력 X) 지표는 LLM 프롬프트로만 전달 ---
                history_compact = [{
                    "q": h["qtext"],
                    "stt": (h.get("stt","")[:150] + "…") if h.get("stt") and len(h["stt"])>150 else (h.get("stt") or "")
                } for h in ss.eva_history[-8:]]
                history_compact_json = json.dumps(history_compact, ensure_ascii=False)

                messages = get_prompt("session").format_messages(
                   voice_summary_json=voice_json,
                posture_summary_json=posture_json,
                history_compact_json=history_compact_json,
                answers=len(ss.get("eva_history", [])),
                )

                summary_text = call_llm(messages)
                st.markdown("## 🧾 면접 결과")
                st.write(summary_text)

                st.success("면접 총평 생성 완료")
                  # ✅ 그리고 다음 렌더에서도 다시 보여줄 수 있게 세션에 저장
                ss.eva_last_summary = summary_text
                ss.eva_show_last_summary = True

            except Exception as e:
                # LLM 실패 시에도 지표 박스는 안 보여주고, 간단 안내만
                st.markdown("## 🧾 면접 총평")
                st.warning("총평 생성에 문제가 발생했어요. 잠시 후 다시 시도해 주세요.")
                st.info(f"(참고: {e})")

            finally:
        # --- 세션 아카이브 (이미 넣은 코드) ---
                ss.setdefault("eva_sessions", []).append({
                    "session_id": ss.session_id,
                    "answers": ss.eva_history.copy(),
                    "summary": summary_text if 'summary_text' in locals() else None,
                    "voice":   locals().get("voice_dict")   or ss.get("eva_voice_summary")   or {},
                    "posture": locals().get("posture_dict") or ss.get("eva_posture_summary") or {},
                    "ended_at": time.time(),
                })
                ss.eva_history = []  # 다음 면접을 위해 비움

                # --- ★ 재시작을 막는 상태값들 리셋 ---
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

                # 🔁 버튼 상태 최신화 위해 강제 한 번 더 렌더
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
