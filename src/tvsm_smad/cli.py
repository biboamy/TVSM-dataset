"""Command-line interface for TVSM speech/music detection."""

from __future__ import annotations

import argparse
import logging
import os

from tvsm_smad import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="tvsm-detect",
        description="Detect speech and music activity in audio files using the TVSM CRNN model.",
    )
    parser.add_argument("--version", action="version", version=f"tvsm-smad {__version__}")
    parser.add_argument("audio_path", help="Path to an audio file or directory of audio files.")
    parser.add_argument("--output-dir", default="output", help="Directory for output CSVs.")
    parser.add_argument(
        "--format",
        choices=["csv", "csv_prob"],
        default="csv",
        help="Output format: 'csv' for binary labels, 'csv_prob' for probabilities.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model-path", default=None, help="Custom model checkpoint path.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from tvsm_smad import SMDetector

    detector = SMDetector(model_path=args.model_path, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.audio_path):
        audio_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(args.audio_path)
            for f in files
            if f.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".ogg"))
        )
    else:
        audio_files = [args.audio_path]

    for audio_file in audio_files:
        basename = os.path.basename(audio_file)
        output_path = os.path.join(args.output_dir, f"{basename}.csv")
        print(f"Processing: {audio_file}")
        detector.predict_to_csv(audio_file, output_path, format_type=args.format)
        print(f"  -> {output_path}")

    print(f"Done. Processed {len(audio_files)} file(s).")
