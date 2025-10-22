import json, os, math
from core.analysis_pose import parse_posture_summary as parse_legacy, normalize_posture
from core.analysis_pose_jetson import parse_posture_summary_jetson
try:
    import numpy as np
except Exception:
    np = None

def _to_native_jsonable(x):
    """np.float32 등 NumPy 스칼라/컬렉션을 파이썬 내장 타입으로 변환."""
    # NumPy 스칼라 → 파이썬 스칼라
    if np is not None and isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, float):
        # NaN/Inf는 None으로 치환 (LLM 입력 안전성)
        if not math.isfinite(x):
            return None
        return float(x)

    if isinstance(x, (int, str, bool)) or x is None:
        return x

    if isinstance(x, dict):
        return {str(k): _to_native_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_native_jsonable(v) for v in x]

    # 그 외(예: set, custom obj)는 문자열화 또는 리스트화 필요 시 확장 가능
    return str(x)

def _severity_by_ratio(count, total):
    if not total: return ""
    r = count / float(total)
    return "강" if r >= 0.40 else "중" if r >= 0.20 else "약" if r >= 0.05 else ""

def _to_common_from_legacy(norm: dict) -> dict:
    frames = int(norm.get("frames_total") or norm.get("frames") or 0)
    neg = int(norm.get("negative_emotion_count", 0))
    bad = int(norm.get("bad_posture_count", 0))
    leg = int(norm.get("leg_shake_count", 0))
    label = norm.get("label") or "정상"
    return {
        "frames": frames,
        "negative_emotion_count": neg,
        "bad_posture_count": bad,
        "leg_shake_count": leg,
        "negative_emotion_severity": _severity_by_ratio(neg, frames),
        "bad_posture_severity": _severity_by_ratio(bad, frames),
        "leg_shake_severity": _severity_by_ratio(leg, frames),
        "label": label,
    }

def parse_posture_auto(xml_path: str) -> tuple[str, dict, dict]:
    """
    return (flavor, common_for_llm, legacy_normalized)
      - flavor: 'jetson' | 'legacy' | 'unknown'
      - common_for_llm: LLM/집계에서 바로 쓰기 좋은 공통 스키마(dict)
            {
              "frames": int,
              "negative_emotion_count": int,  # ← ratio*frames로 근사한 '프레임 수'
              "bad_posture_count": int,
              "leg_shake_count": int,
              "negative_emotion_severity": str,  # 강/중/약 등
              "bad_posture_severity": str,
              "leg_shake_severity": str,
              "label": str,                     # 최종 라벨 문자열
            }
      - legacy_normalized: normalize_posture(...) 결과(레거시일 때만)
    """
    # 1) Jetson 우선 시도
    try:
        jet = parse_posture_summary_jetson(xml_path)
        if jet:

            # ── (A) 새 Jetson 스키마(total_frames / counts / ratios / levels)
            if ("total_frames" in jet) or ("counts" in jet) or ("ratios" in jet):
                frames = int(jet.get("total_frames") or 0)
                ratios = jet.get("ratios") or {}
                levels = jet.get("levels") or {}

                def _est_frames(key: str) -> int:
                    r = ratios.get(key)
                    try:
                        return int(round(float(r) * frames)) if (r is not None and frames > 0) else 0
                    except Exception:
                        return 0

                neg_cnt = _est_frames("negative_emotion")
                bad_cnt = _est_frames("bad_posture")
                leg_cnt = _est_frames("leg_shake")

                # Jetson이 준 등급이 있으면 우선 사용, 없으면 기존 규칙으로 환산
                neg_sev = levels.get("negative_emotion") or _severity_by_ratio(neg_cnt, frames)
                bad_sev = levels.get("bad_posture")      or _severity_by_ratio(bad_cnt, frames)
                leg_sev = levels.get("leg_shake")        or _severity_by_ratio(leg_cnt, frames)

                common = {
                    "frames": frames,
                    "negative_emotion_count": neg_cnt,
                    "bad_posture_count":      bad_cnt,
                    "leg_shake_count":        leg_cnt,
                    "negative_emotion_severity": neg_sev,
                    "bad_posture_severity":      bad_sev,
                    "leg_shake_severity":        leg_sev,
                    "label": jet.get("label") or "정상",
                }
                return "jetson", common, {}

            # ── (B) 구 Jetson 스키마(frames / *_rate / *_severity …)도 그대로 통과
            if isinstance(jet.get("frames"), int):
                return "jetson", jet, {}
    except Exception:
        pass

    # 2) Legacy 폴백
    try:
        raw = parse_legacy(xml_path) or {}
        norm = normalize_posture(raw) or {}
        common = _to_common_from_legacy(norm)
        return "legacy", common, norm
    except Exception:
        pass

    return "unknown", {"frames": 0, "label": "데이터 없음"}, {}


