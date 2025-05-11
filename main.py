import streamlit as st
from accent import download_and_extract_audio, classify_accent

st.title("🎙️ English Accent Classifier")

url = st.text_input("Paste Public Video URL (e.g. Loom or MP4)")

if st.button("Analyze") and url:
    with st.spinner("Processing..."):
        try:
            audio_path = download_and_extract_audio(url)
            accent, score, all_scores, summary = classify_accent(audio_path)

            st.success(f"Accent: **{accent}**")
            st.metric(label="Confidence", value=f"{score}%")
            st.json(all_scores)
            st.info(summary)
        except Exception as e:
            st.error(f"Error: {e}")
