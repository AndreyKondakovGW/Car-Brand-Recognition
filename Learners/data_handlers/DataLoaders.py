import json
import os
import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from data_handlers.DataSets import CustomDataset

def create_loaders(cfg: DictConfig):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = dir_path + cfg.data.path
    print(main_path)
    main_set = CustomDataset(main_path, cfg.data.num_img, cfg.model.params.input_size, cfg.model.params.labels_num)
    
    ind_to_class = {v: k for k, v in main_set.class_d.items()}
    with open("index_to_class.json", "w") as outfile:
        json.dump(ind_to_class, outfile)
    print(len(main_set))
    if (cfg.data.train.path == cfg.data.path):
        train_dataset, val_dataset = split_train_val_ind(main_set, train_split=cfg.data.train.percent)
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    else:
        train_set = CustomDataset(dir_path + cfg.data.train.path, cfg.data.num_img, cfg.model.params.input_size, cfg.model.params.labels_num)
        train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
        val_set = CustomDataset(dir_path + cfg.data.val.path, cfg.data.num_img, cfg.model.params.input_size, cfg.model.params.labels_num)
        val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    
    return train_loader, val_loader


def split_train_val_ind(dataset, train_split=0.75):
    classes = []
    for idx, cl in dataset:
        classes.append(cl)
    train_idx, val_idx, cl_train, cl_test = train_test_split(list(range(len(dataset))), classes, train_size=train_split, stratify=classes)
    
    #print("Train class counts")
    #unique, counts = np.unique(cl_train, return_counts=True)
    #print(np.asarray((unique, counts)).T)
    
    #print("Val class counts")
    #unique, counts = np.unique(cl_test, return_counts=True)
    #print(np.asarray((unique, counts)).T)
    
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    return train, val
