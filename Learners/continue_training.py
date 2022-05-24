import hydra
from omegaconf import DictConfig
from torch import device
from torchsummary import summary
from pytorch_lightning import Trainer

from tools.LightningTrainer2 import LightModuleUpdated
from data_handlers.DataLoaders import create_loaders
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    if cfg.generals.gpus == 0:
        cur_device = device('cpu')
    else:
        cur_device = device('cuda')
    train_loader, val_loader, test_loader = create_loaders(cfg)
    lmodel_finetuned = LightModuleUpdated.load_from_checkpoint(checkpoint_path="../../../pretrain44.ckpt", mode='fine_tune',device = cur_device)
    lmodel_finetuned.model.to(cur_device)
    logger = TensorBoardLogger("tb_logs", name=cfg.model.name)
    summary(lmodel_finetuned.model, (3, cfg.model.params.size, cfg.model.params.size))
    trainer_finetuner= Trainer(logger=logger, accelerator="auto", max_epochs =cfg.data.max_epoch, gpus = cfg.generals.gpus)
    trainer_finetuner.fit(lmodel_finetuned, train_loader, val_loader)



if __name__ == "__main__": 
    my_app()

    
