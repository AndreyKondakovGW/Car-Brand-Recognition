from torchsummary import summary
from torch import device, rand, as_tensor
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import DictConfig
from tools.ModelsConstructor import get_model
from tools.modules.Arcface.arcface_encoder import ArcFaceEncoder
from tools.data.DataLoaders import create_loaders
import os
import random
from tools.LightningTrainer2 import LightModuleUpdated
import json
import pandas as pd


def run_test2(model_path, examples_path, output_path='./output.csv', ind_to_class_json_path='index_to_class.json'):
    directory_path = os.getcwd()
    print(directory_path)
    cur_device = device('cuda')
    preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    with open(ind_to_class_json_path) as json_file:
        ind_to_class_json = json.load(json_file)
        class_to_ind = {v: k for k, v in ind_to_class_json.items()}
        
    lmodel = LightModuleUpdated.load_from_checkpoint(model_path, device= cur_device)
    lmodel.model.to(cur_device)
    lmodel.model.eval()
    
    examples = os.listdir(examples_path)
    res = []
    for ex in examples:
        img_path = examples_path + '/' + ex
        img_tensor = preprocess(Image.open(img_path))
        img_tensor = img_tensor.reshape([1, 3, 224,224]).to(cur_device)
        
        
        emb_x =lmodel.model.gen_embs(img_tensor).cpu().detach().numpy()[0]
        res.append([ex, emb_x.tolist()])
        
    df = pd.DataFrame(data=np.array(res, dtype=object), columns=['name', 'embedding'])
    df.to_csv(output_path)
    
if __name__ == "__main__":
    model_path = './outputs/2022-05-12/00-18-41/outputs/pretrain_arcface2.ckpt'
    ind_to_class_json_path = './outputs/2022-05-12/00-18-41/index_to_class.json'
    examples_path = './data/examples'
    run_test2(model_path, examples_path, ind_to_class_json_path=ind_to_class_json_path)
        
    