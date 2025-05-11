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

    
    embedding = spk_model.encode_batch(waveform)
    embedding = embedding.squeeze(0)  

    
    scores = {}
    for accent, accent_vector in ACCENT_EMBEDDINGS.items():
        embedding_reshaped = embedding.view(1, -1)  
        accent_vector_reshaped = accent_vector.view(1, -1)  
        similarity = torch.nn.functional.cosine_similarity(embedding_reshaped, accent_vector_reshaped, dim=1)
        scores[accent] = similarity.item() * 100  

    best_accent = max(scores, key=scores.get)
    best_score = scores[best_accent]

    summary = f"The model is most confident the accent is {best_accent} with a score of {round(best_score, 2)}%."
    return best_accent, round(best_score, 2), scores, summary