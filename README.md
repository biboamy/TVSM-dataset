# TVSM Dataset

The TV Speech and Music (TVSM) dataset contains speech and music activity labels across a variety of TV shows and their corresponding audio features extracted from professionally-produced high-quality audio. 
The dataset aims to facilitate research on speech and music detection task. 


## Get the dataset

- The dataset can be downloaded via Zenodo.org [URL].
- The paper can be download via EURASIP open access[URL].
- This repo contains materials and codebase to reproduce the baseline experiment in the paper.

## License and attribution
```
@article{TBD,
  title={{A Large TV Dataset for Speech and Music Activity Detection},
  author={Hung, Yun-Ning and Wu, Chih-Wei and Orife, Iroro and Hipple, Aaron and Wolcott, William and Lerch, Alexander},
  journal={EURASIP Journal on Audio, Speech, and Music Processing},
  volume={2022},
  number={TBD},
  pages={TBD},
  year={2022},
  publisher={Springer}
}
```
The TVSM dataset is licensed under a [Apache License 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) 

## Dataset introduction

The downloaded dataset have the following structure:
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
- **TVSM-pseudo/**: larger subset used for trianing. The labels are labeled from a pre-trained model trained on TVSM-cuesheet
- **TVSM-test/**: subset for testing. The labels are labeled by human annotators

Each subset folder has the same structure:
- **labels/**: speech and music activation labels for each sample. Each row in a csv file represents "start time", "end time" and "s(speech)/m(music)"  
- **mel_features/**: the Mel spectrogram feature extracted from the audio of each sample
- **mfcc/**: the MFCCs feature extracted from the audio of each sample
- **vgg_features/**: the [VGGish](https://arxiv.org/pdf/1609.09430.pdf) feature extracted from the audio of each sample
- **TVSM-xxxx_metadata.csv**: the metadata of each sample 

For more information, please visit our paper

## Codebase introduction



## Contact
Please feel free to contact [yunning.hung@bytedance.com](mailto:yunning.hung@bytedance.com) or open an issue here if you have any questions about the dataset or the support code.
