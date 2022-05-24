import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torchsummary import summary
from torch import device

from tools.LightningTrainer2 import LightModuleUpdated
from tools.ModelsConstructor import get_model

from data_handlers.DataLoaders import create_loaders

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    if cfg.generals.gpus == 0:
        cur_device = device('cpu')
    else:
        cur_device = device('cuda')
    
    #loading model
    model = get_model(cfg.model, device=cur_device)
    
    print("Model: ")
    model.to(cur_device)
    summary(model, (3, cfg.model.params.input_size, cfg.model.params.input_size))
    
    train_loader, val_loader = create_loaders(cfg)
    
    
    lmodel_pretrained = LightModuleUpdated(model, mode='train', **cfg.learning_params)
    logger = TensorBoardLogger("tb_logs", name=cfg.model.name, default_hp_metric=False)
    logger_finetuned = TensorBoardLogger("tb_finelogs", name=cfg.model.name, default_hp_metric=False)

    trainer_pre = Trainer(logger=logger, accelerator="auto", max_epochs = cfg.learning_params.max_frozen_epoch, gpus = cfg.generals.gpus)
    trainer_pre.fit(lmodel_pretrained, train_loader, val_loader)
    print("Train acc:")
    trainer_pre.test(dataloaders=train_loader)
    print("Val acc:")
    trainer_pre.test(dataloaders=val_loader)
    trainer_pre.save_checkpoint("outputs/pretrain_" + cfg.model.name +".ckpt")
    
    if (cfg.learning_params.finetuninng):
        print('Warm up all model')
        lmodel_pretrained.warm_up_basemodel()
        finetune_epochs = cfg.data.max_epoch
        trainer_finetuner = Trainer(logger=logger_finetuned, max_epochs = finetune_epochs, gpus = cfg.generals.gpus)
        trainer_finetuner.fit(lmodel_pretrained, train_loader, val_loader, ckpt_path="outputs/pretrain_" + cfg.model.name +".ckpt")
        print("Train acc:")
        trainer_finetuner.test(lmodel_pretrained, dataloaders=train_loader)
        print("Val acc:")
        trainer_finetuner.test(lmodel_pretrained, dataloaders=val_loader)
        trainer_pre.save_checkpoint("outputs/" + cfg.model.name + ".ckpt")

if __name__ == "__main__":
    my_app()