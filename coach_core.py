import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
from faster_whisper import WhisperModel
from openai import OpenAI
from datetime import datetime
import subprocess
import json

from ml.delivery_model import predict_delivery

#python -m streamlit run coach_app.py

# ===== CONFIG =====
FS = 16000
WHISPER_MODEL = "base"  # "base", "small", "medium", "large-v2"
GPT_MODEL = "gpt-4o-mini"
# ==================

client = None

# ===== INITIALIZE OPENAI =====
def init_openai(api_key):
    global client
    client = OpenAI(api_key=api_key)

# ===== RECORD AUDIO =====
def record_audio(file_path, duration=30):
    audio = sd.rec(int(duration * FS), samplerate=FS, channels=1)
    sd.wait()
    write(file_path, FS, audio)

# ===== TRANSCRIBE AUDIO =====
def transcribe_audio(file_path):
    # Load faster‑whisper model
    model = WhisperModel(WHISPER_MODEL, device="cpu")

    # Transcribe
    segments, info = model.transcribe(str(file_path))

    # Build text
    transcript_text = " ".join([segment.text for segment in segments])

    # Convert segments to a compatible format
    result_segments = []
    for i, segment in enumerate(segments):
        result_segments.append({
            "id": i,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    return {
        "text": transcript_text,
        "segments": result_segments,
        "language": info.language
    }

# ===== AUDIO METADATA =====
def get_audio_metadata(file_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=filename,duration,size,format_name:stream=sample_rate,channels",
        "-of", "json",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

# ===== FEATURE EXTRACTION =====
def extract_features(whisper_result, audio_metadata):
    text = whisper_result["text"]
    words = text.split()

    duration = float(audio_metadata["format"]["duration"])
    pace_wpm = (len(words) / duration) * 60 if duration > 0 else 0

    filler_words = ["um", "uh", "like", "you know"]
    filler_count = sum(text.lower().count(w) for w in filler_words)

    avg_pause = duration / max(len(whisper_result["segments"]), 1)

    return [
        pace_wpm,
        filler_count,
        avg_pause,
        duration
    ]

# ===== GPT FEEDBACK =====
def get_speaking_feedback(transcript_text, delivery_score):
    prompt = f"""
You are a public speaking coach.

Analyze the transcript and provide:
- Confidence feedback
- Clarity feedback
- Articulation feedback
- 1–2 improvement suggestions
- Overall rating: Excellent, Good, Needs Improvement, or Poor
- The speaker has a delivery confidence score of {delivery_score:.2f} (0–1 scale).

Transcript:
\"\"\"{transcript_text}\"\"\"
"""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert public speaking coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# ===== BUILD METADATA =====
def build_metadata(file_path, whisper_result):
    transcript_text = whisper_result["text"]
    audio_meta = get_audio_metadata(file_path)
    features = extract_features(whisper_result, audio_meta)

    delivery_score = predict_delivery(features)

    metadata = {
        "file": str(file_path),
        "recording_time": datetime.now().isoformat(),
        "audio_metadata": audio_meta,
        "features": {
            "pace_wpm": features[0],
            "filler_count": features[1],
            "avg_pause": features[2],
            "duration": features[3]
        },
        "ml_scores": {
            "delivery_confidence": round(delivery_score, 2)
        },
        "transcription_metadata": {
            "model": WHISPER_MODEL,
            "language": whisper_result.get("language", "unknown"),
            "num_words": len(transcript_text.split()),
            "num_segments": len(whisper_result["segments"])
        },
        "segments": whisper_result["segments"],
        "transcript": transcript_text,
        "feedback": get_speaking_feedback(transcript_text, delivery_score)
    }
    return metadata

# ===== SAVE METADATA =====
def save_metadata(metadata, file_path):
    out_path = Path(file_path).with_suffix(".metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path