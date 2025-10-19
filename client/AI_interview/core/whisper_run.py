from pathlib import Path
from typing import Optional, Dict, Any
import torch, whisper

def load_whisper(model_size: str = "small") -> whisper.Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    return model

def transcribe_file(
    model: whisper.Whisper,
    audio_path: str | Path,
    *,
    language: str = "ko",
    task: str = "transcribe",
    fp16: Optional[bool] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    if fp16 is None:
        fp16 = torch.cuda.is_available()
    result = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        fp16=fp16,
        temperature=temperature,
        condition_on_previous_text=True,
        verbose=False,
    )
    return result
