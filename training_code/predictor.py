'''
This script is a sample for reference. 
Simply replacing model and hparam path if you want to test on other models.
'''

import argparse
from SM_detector import SM_detector
from utils.general_utils import yaml_to_parser

TVSM_CUESHEET_MODEL_PATH = '../Models/TVSM-cuesheet/epoch=10-step=4058.ckpt'
TVSM_CUESHEET_HPARAM_PATH = '../Models/TVSM-cuesheet/hparams.yaml'
TVSM_PSEUDO_MODEL_PATH = '../Models/TVSM-pseudo/epoch=28-step=67192.ckpt'
TVSM_PSEUDO_HPARAM_PATH = '../Models/TVSM-pseudo/hparams.yaml'


def main(hparam):
    model = SM_detector.load_from_checkpoint(TVSM_PSEUDO_MODEL_PATH, hparam)
    model = model.eval()
    model.prediction(
    	audio_path = hparam.audio_path, 
    	output_dir = hparam.output_dir, 
    	output_name = hparam.output_name)

    return model


if __name__ == '__main__':
    parser = yaml_to_parser(TVSM_PSEUDO_HPARAM_PATH)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--output_name', type=str, default='test')
    hparam = parser.parse_args()
    main(hparam)
