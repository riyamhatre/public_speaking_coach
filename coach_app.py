import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile
import os

#export KMP_DUPLICATE_LIB_OK=TRUE
#python -m streamlit run coach_app.py



from coach_core import (
    init_openai,
    record_audio,
    transcribe_audio,
    build_metadata,
    save_metadata
)

# ===== APP CONFIG =====
st.set_page_config(
    page_title="Public Speaking Coach",
    layout="centered"
)

DATA_DIR = Path("data/recordings")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ===== SIDEBAR =====
st.sidebar.title("🎤 Public Speaking Coach")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your API key is used only for this session"
)

record_duration = st.sidebar.slider(
    "Recording duration (seconds)",
    min_value=10,
    max_value=120,
    value=30
)

if api_key:
    init_openai(api_key)

# ===== MAIN UI =====
st.title("🎙️ Practice Your Speech")

st.markdown("""
Speak naturally as if addressing a real audience.
When you're done, you'll receive **AI-powered coaching feedback**
and **objective delivery scores**.
""")

# ===== RECORD BUTTON =====
if st.button("▶️ Record Speech"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    with st.spinner("Recording... Speak now!"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = DATA_DIR / f"speech_{timestamp}.wav"

        record_audio(audio_path, duration=record_duration)

    st.success("Recording complete!")

    # ===== TRANSCRIBE =====
    with st.spinner("Transcribing speech..."):
        whisper_result = transcribe_audio(audio_path)

    st.subheader("📝 Transcript")
    st.write(whisper_result["text"])

    # ===== ANALYZE & FEEDBACK =====
    with st.spinner("Analyzing delivery and generating feedback..."):
        metadata = build_metadata(audio_path, whisper_result)
        metadata_path = save_metadata(metadata, audio_path)

    # ===== DISPLAY RESULTS =====
    st.subheader("📊 Delivery Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Delivery Confidence",
            metadata["ml_scores"]["delivery_confidence"]
        )

    with col2:
        st.metric(
            "Words per Minute",
            round(metadata["features"]["pace_wpm"], 1)
        )

    st.subheader("🎯 Coaching Feedback")
    st.write(metadata["feedback"])

    st.caption(f"Session saved to: `{metadata_path}`")

# ===== FOOTER =====
st.markdown("---")
st.caption("Powered by Whisper · TensorFlow · MCP · Streamlit")
