"""Tests for the tvsm_smad package."""

import torch

from tvsm_smad.crnn import CRNN
from tvsm_smad.pcen import PCENTransform, F2M


def test_crnn_forward_shape():
    """CRNN should output (batch, 2, time_frames) for a Mel input."""
    model = CRNN()
    model.eval()
    x = torch.randn(1, 128, 625)  # ~20s of Mel frames at sr=16000, hop=512
    with torch.inference_mode():
        out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 2  # music + speech channels


def test_pcen_transform():
    """PCENTransform should accept Mel input and return same shape."""
    transform = PCENTransform()
    x = torch.randn(1, 100, 128)  # (batch, frames, mel_bins)
    out = transform(x, is_mel=True)
    assert out.shape == x.shape


def test_f2m_output_shape():
    """F2M should convert STFT bins to n_mels bins."""
    f2m = F2M(n_mels=128, n_fft=1024)
    x = torch.randn(1, 100, 513)  # (batch, frames, n_fft//2+1)
    out = f2m(x)
    assert out.shape == (1, 100, 128)


def test_bundled_model_exists():
    """The bundled model checkpoint should be resolvable."""
    from tvsm_smad.detector import _default_model_path
    path = _default_model_path()
    assert path.endswith(".pt")


def test_smdetector_init():
    """SMDetector should initialize with the bundled model."""
    from tvsm_smad import SMDetector
    detector = SMDetector()
    assert detector.model is not None
    assert detector.device == torch.device("cpu")
