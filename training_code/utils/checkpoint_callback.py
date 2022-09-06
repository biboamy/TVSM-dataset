import torch
from pytorch_lightning.callbacks import ModelCheckpoint

class BacktrackingModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.old_lr = None
        self.verbase = True

    # get the best model for later re-load
    def get_best_model_path(self):
        if self.mode == 'min':
            return min(self.best_k_models, key=self.best_k_models.get)
        else:
            return max(self.best_k_models, key=self.best_k_models.get)

    # after n epoch no update, load the previous model and re-update
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        current_learning_rate = trainer.optimizers[0].param_groups[0]['lr']

        if self.old_lr is None:
            self.old_lr = current_learning_rate

        new_lr = current_learning_rate
        if new_lr < self.old_lr:
            ckpt_path = self.get_best_model_path()
            ckpt = torch.load(ckpt_path)
            pl_module.load_state_dict(ckpt['state_dict'])
