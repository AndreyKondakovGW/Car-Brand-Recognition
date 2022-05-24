from logging import setLoggerClass
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import Accuracy, Precision, Recall

class LightModuleUpdated(pl.LightningModule):
    '''
        Класс pytorch_lightning LightningModule для удобного обучения модели

        Attributes
        ----------
             - model: pytorch модель
             - criterion:  лосс функция
             - optimizer_ctor: оптимизатор (по умалчанию Адам)
             - lr: скорость обучения
             - weight_decay: l2 регуляризация
    '''

    def __init__(self, 
    model: nn.Module,
    criterion = F.cross_entropy,
    optimizer_ctor = None,
    finetuning_optimizer_ctor = None,
    lr = 1e-3,
    finetuning = False,
    weight_decay = 0.1,
    finetuning_lr = 0.0001,
    mode = 'train',
    **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer_ctor
        self.finetuning_optimizer = finetuning_optimizer_ctor
        self.lr = lr
        self.weight_decay = weight_decay
        self.finetuning_lr = finetuning_lr
        self.finetuning = finetuning

        self.train_acc_metric = Accuracy(compute_on_step=False)
        self.val_acc_metric = Accuracy(compute_on_step=False)
        self.test_acc_metric = Accuracy(compute_on_step=False)

        self.val_recall_metric = Recall(compute_on_step=False)
        self.val_prec_metric = Precision(compute_on_step=False)
        
        self.mode = mode


    def warm_up_basemodel(self):
        self.mode = 'finetune'
        for p in self.model.backbone.parameters():
            p.requires_grad = True


    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        input, target = batch

        output = self.model(input, target)
        
        loss = self.criterion(output, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        predicts = np.argmax(output.cpu().detach().numpy(), axis=1)
        
        self.train_acc_metric(torch.from_numpy(predicts), target.cpu())

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log('train accuracy', self.train_acc_metric,  on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        
        loss = self.criterion(output, target)

        predicts =  np.argmax(output.cpu().detach().numpy(), axis=1)

        self.val_acc_metric(torch.from_numpy(predicts), target.cpu())
        self.val_recall_metric(torch.from_numpy(predicts), target.cpu())
        self.val_prec_metric(torch.from_numpy(predicts), target.cpu())

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log('validation accuracy', self.val_acc_metric,  on_step=False, on_epoch=True, sync_dist=True)
        self.log('validation precision', self.val_recall_metric,  on_step=False, on_epoch=True, sync_dist=True)
        self.log('validation recall', self.val_prec_metric,  on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        
        loss = self.criterion(output, target)

        predicts =  torch.argmax(output, dim=1).detach().cpu().numpy()

        self.test_acc_metric(torch.from_numpy(predicts), target.cpu())
        return loss
    
    def test_epoch_end(self, test_step_outputs):
        self.log('test accuracy', self.test_acc_metric,  on_step=False, on_epoch=True, sync_dist=True)
        
    def _configure_optim_backbone(self):
        if self.optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4, last_epoch=-1), "monitor": "train_loss"}
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler}

    def _configure_optim_finetune(self):
        if self.optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.finetuning_lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1), "monitor": "train_loss"}
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler}



    def configure_optimizers(self):
        print('Choose optimizer')
        print(self.mode)
        if self.mode == 'train': 
            return self._configure_optim_backbone() 
        elif self.mode == 'finetune': 
            return self._configure_optim_finetune()
    