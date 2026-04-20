"""Integration tests for SMDetector predict_audio and predict_to_csv."""

import tempfile
from pathlib import Path

import pytest

from tvsm_smad import SMDetector


def test_predict_audio_integration():
    """Test predict_audio with a real audio file containing speech and music."""
    # Test file path - to be provided
    audio_path = Path(__file__).parent / "fixtures" / "rr_b.rhapsody_test_audio_16kHz.wav"

    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")

    detector = SMDetector()
    results = detector.predict_audio(str(audio_path))

    # Verify output structure
    assert isinstance(results, list)
    assert len(results) > 0

    for r in results:
        assert "start_time_s" in r
        assert "end_time_s" in r
        assert "music_prob" in r
        assert "speech_prob" in r
        assert isinstance(r["start_time_s"], float)
        assert isinstance(r["end_time_s"], float)
        assert 0 <= r["music_prob"] <= 1
        assert 0 <= r["speech_prob"] <= 1
        assert r["end_time_s"] >= r["start_time_s"]


def test_predict_to_csv_integration():
    """Test predict_to_csv with a real audio file."""
    audio_path = Path(__file__).parent / "fixtures" / "rr_b.rhapsody_test_audio_16kHz.wav"

    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")

    detector = SMDetector()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.csv"
        detector.predict_to_csv(str(audio_path), str(output_path), format_type="csv_prob")

        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Check header
        assert lines[0].strip() == "start_time_s,end_time_s,music_prob,speech_prob"

        # Check data rows
        assert len(lines) > 1
        for line in lines[1:]:
            parts = line.split(",")
            assert len(parts) == 4
