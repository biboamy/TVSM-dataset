"""Speech/Music Activity Detector using the TVSM CRNN model.

Example usage::

    from tvsm_smad import SMDetector

    detector = SMDetector()
    results = detector.predict_audio("track.wav")

    for r in results:
        print(f"{r['start_time_s']:.2f}-{r['end_time_s']:.2f}  "
              f"music={r['music_prob']:.2f}  speech={r['speech_prob']:.2f}")
"""

from __future__ import annotations

import csv
import logging
from importlib import resources

import librosa
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T

from tvsm_smad.crnn import CRNN
from tvsm_smad.pcen import PCENTransform

logger = logging.getLogger(__name__)

SR = 16000
N_FFT = 1024
HOP_SIZE = 512
CHUNK_DURATION_S = 20

_BUNDLED_MODEL = "models/TVSM-pseudo/epoch=28-step=67192.ckpt.torch.pt"


def _default_model_path() -> str:
    """Resolve the path to the bundled model checkpoint."""
    ref = resources.files("tvsm_smad").joinpath(_BUNDLED_MODEL)
    with resources.as_file(ref) as p:
        return str(p)


class SMDetector:
    """Speech/Music detector backed by a TVSM CRNN checkpoint.

    Args:
        model_path: Path to a .pt state-dict checkpoint.
            If None, uses the bundled TVSM-pseudo model.
        device: Torch device string ("cpu", "cuda", "mps").
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CRNN()

        if model_path is None:
            model_path = _default_model_path()

        logger.info("Loading TVSM model from %s", model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.pcen_transform = T.Compose(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=SR,
                    n_fft=N_FFT,
                    hop_length=HOP_SIZE,
                    n_mels=128,
                ).to(self.device),
                PCENTransform().to(self.device),
            ]
        )
        logger.info("TVSM detector ready on %s", self.device)

    def predict_audio(self, audio_path: str) -> list[dict]:
        """Run inference on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of dicts with keys:
                - start_time_s (float)
                - end_time_s (float)
                - music_prob (float, 0-1)
                - speech_prob (float, 0-1)
        """
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = np.expand_dims(y, 0)
        audio = torch.from_numpy(y).float().to(self.device)

        audio_pcen = self.pcen_transform(audio)
        chunk_frames = int(SR / HOP_SIZE * CHUNK_DURATION_S)
        n_chunks = int(np.ceil(audio_pcen.shape[-1] / chunk_frames))

        chunks = []
        with torch.inference_mode():
            for i in range(n_chunks):
                chunk = audio_pcen[..., i * chunk_frames : (i + 1) * chunk_frames]
                out = self.model(chunk).detach().cpu()
                chunks.append(out)

        est_label = torch.cat(chunks, -1)
        est_label = torch.sigmoid(est_label)
        est_label = torch.max_pool1d(est_label, 6, 6)

        frame_time = 1 / ((SR / HOP_SIZE) / 6)
        est_np = est_label.detach().cpu().numpy()[0]

        results = []
        for i, frame in enumerate(est_np.T):
            results.append(
                {
                    "start_time_s": float(frame_time * i),
                    "end_time_s": float(frame_time * (i + 1)),
                    "music_prob": round(float(frame[0]), 4),
                    "speech_prob": round(float(frame[1]), 4),
                }
            )

        return results

    def predict_to_csv(
        self,
        audio_path: str,
        output_path: str,
        music_threshold: float = 0.5,
        speech_threshold: float = 0.5,
        format_type: str = "csv",
    ) -> None:
        """Run inference and write results to a CSV file.

        Args:
            audio_path: Path to the audio file.
            output_path: Path for the output CSV.
            music_threshold: Threshold for music activation.
            speech_threshold: Threshold for speech activation.
            format_type: "csv" for binary labels, "csv_prob" for probabilities.
        """
        results = self.predict_audio(audio_path)

        with open(output_path, "w", newline="") as f:
            if format_type == "csv":
                for r in results:
                    if r["music_prob"] > music_threshold:
                        f.write(f"{r['start_time_s']}\t{r['end_time_s']}\tm\n")
                    if r["speech_prob"] > speech_threshold:
                        f.write(f"{r['start_time_s']}\t{r['end_time_s']}\ts\n")
            elif format_type == "csv_prob":
                writer = csv.DictWriter(
                    f, fieldnames=["start_time_s", "end_time_s", "music_prob", "speech_prob"]
                )
                writer.writeheader()
                for r in results:
                    writer.writerow(r)

        logger.info("Wrote results to %s", output_path)
