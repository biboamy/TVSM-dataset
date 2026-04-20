# TVSM Dataset

The TV Speech and Music (TVSM) dataset contains speech and music activity labels across a variety of TV shows and their corresponding audio features extracted from professionally-produced high-quality audio. 
The dataset aims to facilitate research on speech and music detection tasks. 


## Get the dataset

- The dataset can be downloaded via [Zenodo.org](https://zenodo.org/record/7025971).
- The paper can be downloaded via [EURASIP open access](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-022-00253-8).
- This repo contains materials and codebase to reproduce the baseline experiment in the paper.

## License and attribution
```
@ARTICLE{Hung2022,
  title={A Large TV Dataset for Speech and Music Activity Detection},
  author={Hung, Yun-Ning and Wu, Chih-Wei and Orife, Iroro and Hipple, Aaron and Wolcott, William and Lerch, Alexander},
  journal={EURASIP Journal on Audio, Speech, and Music Processing},
  volume={2022},
  number={1},
  pages={21},
  year={2022},
  publisher={Springer}
}
```
The TVSM dataset is licensed under a [Apache License 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) 

## Dataset introduction

The downloaded dataset has the following structure:
```
└─── READEME.txt
└─── TVSM-cuesheet/
│    └─── labels/
│    └─── mel_features/
│    └─── mfcc/
│    └─── vgg_features/
│    └─── TVSM-xxxx_metadata.csv
└─── TVSM-pseudo/
└─── TVSM-test/
```

- **READEME.txt**: basic information about the dataset
- **TVSM-cuesheet/**: smaller subset used for training. The labels are derived from cuesheet information
- **TVSM-pseudo/**: larger subset used for training. The labels are labeled from a pre-trained model trained on TVSM-cuesheet
- **TVSM-test/**: subset for testing. The labels are labeled by human annotators

Each subset folder has the same structure:
- **labels/**: speech and music activation labels for each sample. Each row in a csv file represents "start time", "end time" and "s(speech)/m(music)"  
- **mel_features/**: the Mel spectrogram feature extracted from the audio of each sample
- **mfcc/**: the MFCCs feature extracted from the audio of each sample
- **vgg_features/**: the [VGGish](https://arxiv.org/pdf/1609.09430.pdf) feature extracted from the audio of each sample
- **TVSM-xxxx_metadata.csv**: the metadata of each sample 

For more information, please visit our paper

## Inference Code Packaging

- Provide a **pip-installable** inference package so users do not need to clone the repo, wire `PYTHONPATH`, or hunt for checkpoints on Google Drive.
- Keep the **training code, evaluation outputs, and legacy `inference/` scripts** in the repository unchanged for reproducibility; the PyPI surface is **inference-only**.
- Bundle the **TVSM-pseudo** converted checkpoint (~3.2 MB) so `pip install` works offline after install.

### Naming (consistent across layers)

| Layer             | Name          | Notes                                                                                                |
|-------------------|---------------|------------------------------------------------------------------------------------------------------|
| **PyPI / pip**    | `tvsm-smad`   | Hyphen is normal for distribution names. Install: `pip install tvsm-smad`.                           |
| **Python import** | `tvsm_smad`   | Underscores only (hyphens are invalid in import paths). Example: `from tvsm_smad import SMDetector`. |
| **CLI**           | `tvsm-detect` | Entry point defined in `pyproject.toml` under `[project.scripts]`.                                   |

The PyPI name `smad` alone was already taken; **`tvsm-smad`** ties the dataset/repo identity (TVSM) to the task (SMAD).

### PyPI Package layout

```
└─── src/tvsm_smad/
    └─── __init__.py      # exports SMDetector, __version__
    └─── crnn.py          # CRNN architecture
    └─── pcen.py          # Mel + PCEN preprocessing
    └─── detector.py      # SMDetector, bundled model resolution, predict_audio / predict_to_csv
    └─── cli.py           # tvsm-detect CLI
    └─── py.typed         # PEP 561 marker for type checkers
    └─── models/
        └─── TVSM-pseudo/
            └─── epoch=28-step=67192.ckpt.torch.pt   # bundled weights (package_data)
```

- **`pyproject.toml`**: build metadata, dependencies (`torch`, `torchaudio`, `librosa`, `numpy`), optional `[dev]`, and `package-data` for `models/**/*.pt`.
- **`tests/`**: pytest tests for shapes, bundled model path, and `SMDetector` init.

---
### Previous Inference Code Structure
Thanks @owlwang for the contribution! The easy-to-use inference code is now included in `inference/`

- **`inference/`** (original scripts) remains for backward compatibility and paper reproduction; it is **not** replaced by the package.
- **`training_code/`**, **`Models/`**, **`Evaluation_Output/`** are unchanged.
- Users who prefer the old workflow can keep using `inference/inference.py` after cloning.
```
cd inference
python3 inference.py --audio_path test.wav --output_dir output/ --format csv/csv_prob
```


### Older inference code

**Interested in inferencing existing samples? Please visit [predictor.py](https://github.com/biboamy/TVSM-dataset/blob/master/training_code/predictor.py) for usage.**

```
cd training_code
python3 predictor.py --audio_path test.wav
```

Please install [git lfs](https://git-lfs.com/) first then run `git-lfs pull` to restore the checkpoints

Please replace `line 31` in `SM_detector.py` with `self.save_hyperparameters(hparams)` if you are using newer pytorch_lightning versions.

```
└─── Evaluation_Output/
│    └─── AVASpeech/
│    │    └─── T2
│    │    └─── TVSM-cuesheet
│    │    └─── TVSM-pseudo
│    └─── ...
└─── Models/
└─── training_code/
```

- **Evaluation_Output**: the output generated by three models across five evaluation sets
  - T2: baseline method  
  - TVSM-cuesheet: CRNN-P-Cue method  
  - TVSM-pseudo: CRNN-P-Pseu method  
- **Models**: the pre-trained checkpoint from CRNN-P-Cue and CRNN-P-Pseu methods
- **training_code**: code for training the model

## Testing (for developers)

This section is for developers contributing to the `tvsm-smad` package. If you're using the package for inference only, see the "Inference Code Packaging" section above.

**Prerequisites:** Python 3.10 or higher is required (the package supports Python 3.10–3.13).

1. Set up a virtual environment (recommended):
   ```bash
   python3.10 -m venv smad_venv
   source smad_venv/bin/activate
   ```

2. Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite with pytest:
   ```bash
   pytest -v
   ```

   Note: Tests require the bundled TVSM-pseudo model checkpoint (~3MB), which is included in the repo at `src/tvsm_smad/models/TVSM-pseudo/`. If you've cloned the repo, the model should already be available.

3. Verify the CLI is installed:
   ```bash
   tvsm-detect --version
   ```

## Bug Fix
If you encounter error "**batch response: This repository is over its data quota. Account responsible for LFS...**", 
please download the model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1THtEHYUh1lueUFH37n2VAhVy8n2QfNpp?usp=sharing)

## Contact
Please feel free to contact [yhung33@gatech.edu](mailto:yhung33@gatech.edu) or open an issue here if you have any questions about the 
dataset or the support code.
