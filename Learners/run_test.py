from torch import device
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from tools.LightningTrainer2 import LightModuleUpdated
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def run_test(model_path, data_path, train_p = 0.7, ind_to_class_json_path='index_to_class.json'):
    directory_path = os.getcwd()
    print(directory_path)
    cur_device = device('cuda')

    
    with open(ind_to_class_json_path) as json_file:
        ind_to_class_json = json.load(json_file)
        class_to_ind = {v: k for k, v in ind_to_class_json.items()}
    
    lmodel = LightModuleUpdated.load_from_checkpoint(model_path, device= cur_device)
    lmodel.model.to(cur_device)
    lmodel.model.eval()
    
    classes = os.listdir(data_path)
    acc_sum = 0
    acc_sum_c = 0
    for cl in random.sample(classes, k=200):
        class_path = data_path + '/' + cl
        images = os.listdir(class_path)
        print(f"Класс {cl}")
        train_images = random.sample(images, k=int(len(images) * train_p))
        train_embs = create_train_embs(lmodel, class_path,  train_images, cur_device)
        centroid = np.mean(train_embs, axis=0)
        mean_cos_dist = np.mean(cosine_similarity(train_embs))
        #print(mean_cos_dist)
            
        total_num = len(images)
        num_correct = 0
        num_correct_centroid = 0
        for img in images:
            img_path = class_path + '/' + img
            emb_x = gen_embedding(lmodel, img_path, cur_device)
            dists = []
            centroid_dist = cosine_similarity([centroid], [emb_x])
            if centroid_dist > mean_cos_dist:
                num_correct_centroid += 1

            for t_e in train_embs:
                dist = cosine_similarity([t_e], [emb_x])
                dists.append(dist)
                
            if np.mean(dists) > mean_cos_dist:
                num_correct += 1
        acc = num_correct / total_num
        acc_centroid = num_correct_centroid / total_num
        acc_sum += acc
        acc_sum_c += acc_centroid
        print(f"Точность: {acc}")
        print(f"Точность по центроиде: {acc_centroid}")
    print(f"Средняя точность: {acc_sum / 200}")
    print(f"Средняя точность по центроиде: {acc_sum_c / 200}") 

def create_train_embs(model, class_path,  samples, cur_device):
    res = []
    for im in samples:
        emb_x = gen_embedding(model, class_path + '/' + im, cur_device)
        res.append(emb_x)
    return np.array(res)

def gen_embedding(model, image_path, cur_device):
    img_tensor = preprocess(Image.open(image_path))
    img_tensor = img_tensor.reshape([1, 3, 224,224]).to(cur_device)
    emb_x =model.model(img_tensor).cpu().detach().numpy()[0].tolist()
    
    return emb_x

if __name__ == "__main__":
    model_path = './outputs/2022-05-20/23-27-10/outputs/embedding_model.ckpt'
    ind_to_class_json_path = './outputs/2022-05-20/23-27-10/index_to_class.json'
    data_path = './data/images_balance'
    run_test(model_path, data_path, ind_to_class_json_path=ind_to_class_json_path)
    
    