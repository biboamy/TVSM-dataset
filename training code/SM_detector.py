import pytorch_lightning as pl
import torch
import random
from loguru import logger
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import os
from utils.creator import dataset_creator, model_creator, preprocess_creator
from utils.general_utils import model_output_to_csv
from utils.ctc_loss import ctl_loss
import soundfile as sf
import torchaudio

EPSILON = 10e-12

class SM_detector(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.transform = preprocess_creator(hparams, 'melspec')
        self.model = model_creator(hparams, hparams.n_features, hparams.n_class, hparams.model_name)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.loss_func = nn.functional.binary_cross_entropy_with_logits
        self.ctc_loss = ctl_loss

        if hparams.use_pcen:
            self.transform_pcen = preprocess_creator(hparams, 'pcen')
        self.use_pcen = hparams.use_pcen

        self.hparams = hparams
        self.use_device = str("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch, partition, opt_idx=0):
        oup_dict = {}
        loss = 0

        # for training
        if partition != 'te':
            x, label, name = batch
            if self.hparams.dataset == 'netflix_whole' and partition == 'tr':
                x = x.squeeze(0)
                label = label.squeeze(0)
        # for testing
        else:
            audio, label, name = batch

        # change the input audio learning rate to 16000
        if partition == 'te':
            if self.hparams.dataset == 'netflix_whole':
                x = audio
            else:
                resample = torchaudio.transforms.Resample(22050, 16000)
                audio = resample(audio)
                # extract log mel spectrogram
                x = self.transform(audio.unsqueeze(1).float()).squeeze(1)[:, :self.hparams.n_features]
            
        # activation estimation
        if self.use_pcen:
            x = self.transform_pcen(x)
        else:
            x = torch.log(x + EPSILON)
        length = min(x.shape[-1], label.shape[-1])
        if partition == 'te':
            c_size = int(x.shape[-1] // 4)
            y = []
            for i in range(x.shape[-1]//c_size):
                y.append(self.model(x[..., i*(c_size):(i+1)*c_size]))
            y = torch.cat(y, -1)[..., :length]
        else:
            y = self.model(x)[..., :length]
        label = label[..., :length]

        # ctl
        if partition == 'tr' and self.hparams.use_ctc:
            # get label onset and offset
            label_ctc = torch.max_pool1d(label, 20, 20)
            y_ctc = torch.max_pool1d(y, 20, 20)
            roll_r = torch.roll(label_ctc, 1, -1)
            roll_l = torch.roll(label_ctc, -1, -1)
            roll_r[..., 0] = 0
            roll_l[..., -1] = 0
            onset = label_ctc - roll_r
            offset = label_ctc - roll_l
            onset[onset < 0] = 0
            offset[offset < 0] = 0
            onset[:, 1] = onset[:, 1] * 3
            offset[:, 0] = offset[:, 0] * 2
            offset[:, 1] = offset[:, 1] * 4
            labels = torch.max(torch.cat([onset, offset], 1), 1)[0]
            ctc_label = []

            seqLen = torch.full((y_ctc.shape[0],), y_ctc.shape[-1], dtype=torch.long).to(self.use_device)

            label_legnth = []
            for i, l in enumerate(labels):
                overlap_start_index = torch.where(onset[i].sum(0) == 4)
                overlap_end_index = torch.where(offset[i].sum(0) == 6)
                if len(overlap_start_index[0])>0:
                    l = torch.cat([torch.full((1,), 1, dtype=torch.long).to(self.use_device), l])
                if len(overlap_end_index[0])>0:
                    l = torch.cat([l, torch.full((1,), 2, dtype=torch.long).to(self.use_device)])
                l = l[l > 0]
                ctc_label.append(l)
                label_legnth.append(len(l))
            for i, l in enumerate(ctc_label):
                ctc_label[i] = (torch.nn.functional.pad(l, (0, max(label_legnth) - len(l))).unsqueeze(0))

            label_legnth = torch.from_numpy(np.array(label_legnth)).to(self.use_device)

            # for CMU ctl
            if self.hparams.ctc_type == 'CMU_ctl':
                # class = 2
                loss_ctc = self.ctc_loss(torch.sigmoid(y_ctc).permute(0, 2, 1), seqLen, torch.cat(ctc_label, 0).type(torch.int64)-1, label_legnth, maxConcur=2) * self.hparams.ctc_weight
            # for CMU ctc
            elif self.hparams.ctc_type == 'CMU_ctc':
                # class = 5
                loss_ctc = self.ctc_loss(y_ctc.permute(0, 2, 1).log_softmax(2), seqLen, torch.cat(ctc_label, 0).type(torch.int64), label_legnth) * self.hparams.ctc_weight
            else:
                # for pytorch ctc
                loss_ctc = self.ctc_loss(y_ctc.permute(2, 0, 1).log_softmax(2), torch.cat(ctc_label, 0).to(self.use_device), seqLen, label_legnth) * self.hparams.ctc_weight

            oup_dict[partition + '_ctc_loss'] = loss_ctc
            loss += loss_ctc

        y = torch.max_pool1d(y, self.hparams.pool_size, self.hparams.pool_size)
        label = torch.max_pool1d(label, self.hparams.pool_size, self.hparams.pool_size)
        # calculate loss if not testing
        if partition != 'te':
            loss_act = self.loss_func(y[..., :length], label[..., :length])
            oup_dict[partition + '_act_loss'] = loss_act
            loss += loss_act

        # decrease the output resolution
        if partition == 'te':
            y = torch.max_pool1d(y, 3, 3)

        return loss, oup_dict, [y[..., :length], label[..., :length], name]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, oup_dict, _ = self.forward(batch, 'tr', optimizer_idx)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, oup_dict, _ = self.forward(batch, 'va')
        self.log_dict(oup_dict)
        return oup_dict

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        key_list = outputs[0].keys()
        for k in key_list:
            tqdm_dict[k] = torch.stack([x[k] for x in outputs]).mean()
        logger.info('Epoch {}, Activation Loss, {:.6f}'.format(self.trainer.current_epoch + 1, tqdm_dict['va_act_loss']))
        self.log('val_loss', tqdm_dict['va_act_loss'], prog_bar=True)
        print()
        #return {'val_loss': tqdm_dict['va_act_loss'], 'log': tqdm_dict}

    def test_step(self, batch, batch_idx):
        _, _, [est_label, _, name] = self.forward(batch, 'te')

        des_path = os.path.join('./evaluation/', self.hparams.model_name, self.hparams.dataset, self.hparams.version)
        frame_time = 1 / ((self.hparams.sr / self.hparams.hop_size) / self.hparams.pool_size / 3)
        est_label = torch.sigmoid(est_label)

        est_label[:, 0][est_label[:, 0] > self.hparams.threshold_m] = 1
        est_label[:, 0][est_label[:, 0] <= self.hparams.threshold_m] = 0
        est_label[:, 1][est_label[:, 1] > self.hparams.threshold_s] = 1
        est_label[:, 1][est_label[:, 1] <= self.hparams.threshold_s] = 0
        est_label = est_label.detach().cpu().numpy()[0]
        print(est_label.shape, frame_time, name[0])
        model_output_to_csv(est_label.T, frame_time, des_path, os.path.basename(name[0]))
        return {'m': est_label[0].sum(), 's': est_label[1].sum(), 'l': est_label.shape[-1]}

    def test_epoch_end(self, outputs):
        oup_dict = {'Music': [], 'Speech': []}
        length = 0
        for output in outputs:
            oup_dict['Music'].append(output['m'])
            oup_dict['Speech'].append(output['s'])
            length += output['l']
        oup_dict['Music'] = sum(oup_dict['Music']) / length
        oup_dict['Speech'] = sum(oup_dict['Speech']) / length
        return oup_dict

    def configure_optimizers(self):

        optimizer_det = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler_det = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_det, patience=self.hparams.patience, factor=self.hparams.decay_factor, verbose=True)

        return {'optimizer': optimizer_det, 'lr_scheduler': scheduler_det, 'monitor': 'val_loss'}

    def _set_seed(self, worker_id):
        random.seed(self.hparams.seed+worker_id)
        np.random.seed(self.hparams.seed+worker_id)
        torch.manual_seed(self.hparams.seed+worker_id)

    def _get_data_loader(self, partition):
        batch_size = self.hparams.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False
        d = dataset_creator(self.hparams, partition)
        return data.DataLoader(d, batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers, worker_init_fn=self._set_seed, drop_last=False)

    def train_dataloader(self):
        return self._get_data_loader('train')

    def val_dataloader(self):
        return self._get_data_loader('val')

    def test_dataloader(self):
        return self._get_data_loader('test')

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params, dataset=None):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        import yaml
        from argparse import Namespace
        default_params = yaml.safe_load(open(default_params))
        inp_harams = Namespace(**default_params)

        model = cls(inp_harams)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if dataset is not None:
            model.hparams.dataset = dataset
            model.hparams.data_path = '../data/' + dataset

        return model