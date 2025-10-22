
import argparse
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any

def _parse_tiers(s: str, default: Tuple[float, float]) -> Tuple[float, float]:
    try:
        a, b = [float(x.strip()) for x in s.split(",")]
        if a <= 0 or b <= 0 or b < a:
            raise ValueError
        return (a, b)
    except Exception:
        return default

def _severity_2level(ratio: float, tiers: Tuple[float, float]) -> str:
    mild, strong = tiers  # e.g., (0.05, 0.25)
    if ratio >= strong:
        return "매우 나쁘다"
    elif ratio >= mild:
        return "약간 나쁘다"
    else:
        return "정상"

def _estimate_total_frames(events):
    # Prefer actual frame span if any frame attribute exists
    frames = [f for (_, f, _) in events if f is not None]
    if frames:
        return max(frames) - min(frames) + 1
    return None  # Unknown

def parse_posture_summary_jetson(xml_path: str,
                          frames_per_log: int = 10,
                          neg_tiers: Tuple[float, float] = (0.05, 0.25),
                          bad_tiers: Tuple[float, float] = (0.05, 0.25),
                          leg_tiers: Tuple[float, float] = (0.05, 0.25)) -> Dict[str, Any]:
    """
    Compute session-level severities using *overall frame ratios*.
    ratio = (event_count * frames_per_log) / total_frames
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []
    neg = bad = leg = 0
    min_frame = None
    max_frame = None

    for ev in root.findall("event"):
        t = (ev.get("type") or "").strip()
        if t not in ("negative_emotion", "bad_posture", "leg_shake"):
            continue
        f = None
        if ev.get("frame") is not None:
            try:
                f = int(float(ev.get("frame")))
            except Exception:
                f = None
        if t == "negative_emotion":
            neg += 1
        elif t == "bad_posture":
            bad += 1
        elif t == "leg_shake":
            leg += 1

        events.append((t, f, ev.get("timestamp_sec")))
        if f is not None:
            min_frame = f if (min_frame is None or f < min_frame) else min_frame
            max_frame = f if (max_frame is None or f > max_frame) else max_frame

    total_frames = _estimate_total_frames(events)
    # Fallback: approximate from the category with most events, when frames missing
    if total_frames is None:
        approx_span = max(neg, bad, leg) * frames_per_log
        total_frames = approx_span if approx_span > 0 else 0

    # Avoid division by zero
    if total_frames <= 0:
        neg_ratio = bad_ratio = leg_ratio = 0.0
    else:
        neg_ratio = (neg * frames_per_log) / total_frames
        bad_ratio = (bad * frames_per_log) / total_frames
        leg_ratio = (leg * frames_per_log) / total_frames

    neg_level = _severity_2level(neg_ratio, neg_tiers)
    bad_level = _severity_2level(bad_ratio, bad_tiers)
    leg_level = _severity_2level(leg_ratio, leg_tiers)

    label = ", ".join([
        f"표정:{neg_level}",
        f"자세:{bad_level}",
        f"다리 떨림:{leg_level}",
    ])

    return {
        "frames_per_log": frames_per_log,
        "total_frames": total_frames,
        "counts": {
            "negative_emotion": neg,
            "bad_posture": bad,
            "leg_shake": leg,
        },
        "ratios": {
            "negative_emotion": round(neg_ratio, 4),
            "bad_posture": round(bad_ratio, 4),
            "leg_shake": round(leg_ratio, 4),
        },
        "levels": {
            "negative_emotion": neg_level,
            "bad_posture": bad_level,
            "leg_shake": leg_level,
        },
        "label": label,
        "frame_span": (min_frame, max_frame) if (min_frame is not None and max_frame is not None) else None,
    }

def _cli():
    ap = argparse.ArgumentParser(description="Session severity by overall frame ratios (2-level).")
    ap.add_argument("-i", "--input", required=True, help="XML path (session events)")
    ap.add_argument("--frames-per-log", type=int, default=10, help="Stride used when logging events (default 10)")
    ap.add_argument("--neg-tiers", default="0.10,0.25", help="mild,strong ratio for negative_emotion (default 0.05,0.25)")
    ap.add_argument("--bad-tiers", default="0.10,0.25", help="mild,strong ratio for bad_posture (default 0.05,0.25)")
    ap.add_argument("--leg-tiers", default="0.10,0.25", help="mild,strong ratio for leg_shake (default 0.05,0.25)")
    args = ap.parse_args()

    neg_t = _parse_tiers(args.neg_tiers, (0.05, 0.25))
    bad_t = _parse_tiers(args.bad_tiers, (0.05, 0.25))
    leg_t = _parse_tiers(args.leg_tiers, (0.05, 0.25))

    res = parse_posture_summary_jetson(args.input, frames_per_log=args.frames_per_log,
                                neg_tiers=neg_t, bad_tiers=bad_t, leg_tiers=leg_t)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _cli()
