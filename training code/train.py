import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import random
import glob
import os
import numpy as np
from utils.general_utils import yaml_to_parser
from SM_detector import SM_detector
from SM_vgg_detector import SMVggDetector
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from utils.checkpoint_callback import BacktrackingModelCheckpoint
from utils.general_utils import seed_everything
from utils.early_stopping import TrackEarlyStopping
import subprocess as sp

def main(hparams):
    seed_everything()

    logger = TensorBoardLogger(save_dir=hparams.save_path)
    hparams.save_path = os.path.join(hparams.save_path, logger.name, f'version_{logger.version}')

    early_stop_callback = TrackEarlyStopping(patience=hparams.early_stopping_patience, verbose=True, mode='min', monitor='val_loss')
    ckpt_path = os.path.join(hparams.save_path, 'ckeckpoints')
    ckpt_callback = BacktrackingModelCheckpoint(ckpt_path)

    if hparams.model_name == 'tcn' or hparams.model_name == 'crnn':
        model = SM_detector(hparams)
    elif hparams.model_name == 'vgg':
        model = SMVggDetector(hparams)
    if hparams.load_pretrain:
        model = model.load_from_checkpoint(glob.glob(f'logger/default/version_{hparams.load_pretrain}/ckeckpoints/*.ckpt')[0], default_params='./parameters.yaml')
    trainer = Trainer(min_epochs=hparams.epochs, max_epochs=hparams.epochs, logger=logger, progress_bar_refresh_rate=1, gradient_clip_val=hparams.gradient_clip_val,
                      gpus=hparams.gpus, weights_save_path=hparams.save_path, checkpoint_callback=ckpt_callback, callbacks = [early_stop_callback])
    trainer.fit(model)

    ffmpeg_command = f'python test.py --ckpt_path {ckpt_path} --target {hparams.target}'
    sp.call(ffmpeg_command, shell=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args, other = parser.parse_known_args()
    hparam = yaml_to_parser('parameters.yaml')
    hparam = hparam.parse_args(other)
    main(hparam)
