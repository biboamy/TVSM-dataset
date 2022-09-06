from pytorch_lightning.callbacks import EarlyStopping
import torch
import numpy as np
from loguru import logger

class TrackEarlyStopping(EarlyStopping):

    # set the early stopping parameters
    def on_train_start(self, trainer, pl_module):
        torch_inf = torch.tensor(np.Inf)
        self.wait = 0
        self.stopped_epoch = 0
        self.best = torch_inf if self.monitor_op == torch.lt else -torch_inf

    # check if the early stopping requirement is reached
    def _run_early_stopping_check(self, trainer, pl_module):
        logs = trainer.callback_metrics
        stop_training = False
        if not self._validate_condition_metric(logs):
            return stop_training
        current = logs.get(self.monitor)

        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                stop_training = True
                self.on_train_end(trainer, pl_module)
        return stop_training

    # if early stopping requirement fulfilled, then quit the training
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(f'Epoch {self.stopped_epoch }: early stopping')
            quit()
