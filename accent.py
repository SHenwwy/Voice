import torch
import torchaudio
import os
import tempfile
import requests
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition

spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

ACCENT_EMBEDDINGS = {
    "American": torch.randn(384),
    "British": torch.randn(384),
    "Australian": torch.randn(384),
    "Indian": torch.randn(384)
}


def load_and_preprocess(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Check audio length (must be at least ~1 second)
    duration_seconds = waveform.shape[1] / sample_rate
    if duration_seconds < 1:
        raise ValueError("Audio is too short to process.")

    return waveform, sample_rate



def download_and_extract_audio(url: str) -> str:
    response = requests.get(url)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(response.content)
    temp_video.close()

    audio_path = temp_video.name.replace(".mp4", ".wav")
    AudioSegment.from_file(temp_video.name).export(audio_path, format="wav")
    return audio_path


def classify_accent(audio_path):
    waveform, sample_rate = load_and_preprocess(audio_path)

    # Add batch dimension
    embedding = spk_model.encode_batch(waveform.unsqueeze(0))
    
    # Validate output
    if embedding is None or embedding.ndim != 3:
        raise ValueError("Failed to generate valid speaker embedding.")

    embedding = embedding.squeeze().detach()

    scores = {}
    for accent, accent_vector in ACCENT_EMBEDDINGS.items():
        similarity = torch.nn.functional.cosine_similarity(
            embedding.view(1, -1), accent_vector.view(1, -1), dim=1
        )
        scores[accent] = similarity.item() * 100

    best_accent = max(scores, key=scores.get)
    best_score = scores[best_accent]

    summary = f"The model is most confident the accent is {best_accent} with a score of {round(best_score, 2)}%."
    return best_accent, round(best_score, 2), scores, summary
