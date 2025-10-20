import opensmile
from typing import Tuple, Dict

# openSMILE 초기화 (eGeMAPS Functionals)
_smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def analyze_stability(audio_path: str) -> Tuple[Dict, str]:
    row = _smile.process_file(audio_path).iloc[0]
    features = {
        "jitter":       row["jitterLocal_sma3nz_amean"],
        "shimmer":      row["shimmerLocaldB_sma3nz_amean"],
        "hnr":          row["HNRdBACF_sma3nz_amean"],
        "f0_std":       row["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
        "loudness_std": row["loudness_sma3_stddevNorm"],
    }
    return features, "ok"

def get_stability_score(jitter: float, shimmer: float, hnr: float):
    JITTER_REF = 0.07
    SHIMMER_REF = 0.6
    score_jitter = max(0, 1 - (jitter / JITTER_REF))
    score_shimmer = max(0, 1 - (shimmer / SHIMMER_REF))
    score_hnr = min(1.0, max(0, hnr / 20.0))
    score = (score_jitter * 0.25 + score_shimmer * 0.25 + score_hnr * 0.5) * 10
    score = min(10.0, (score * 1.4) + 2.0)
    if score >= 8.0:  label, color = "안정적 ✅", "success"
    elif score >= 5.0: label, color = "보통 ⚠️", "warning"
    else:              label, color = "불안정 ❌", "error"
    return round(score, 2), label, color


