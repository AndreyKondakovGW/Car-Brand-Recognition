import glob
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from omegaconf import DictConfig
from PIL import ImageFile
import random
FULL_DATASET_SIZE = 100
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_datasets(cfg: DictConfig):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_base = CustomDataset(dir_path + cfg.data.path, cfg.data.size)

class CustomDataset(Dataset):
    def __init__(self, path, num_img, img_size, class_num = FULL_DATASET_SIZE, train=True):
        #Если train=True то отбрасываем все классы с кол вом картинок меньше num_img
        #Иначе только их берём
        self.imgs_path = path
        
        if class_num < FULL_DATASET_SIZE:
            class_list = random.sample(glob.glob(self.imgs_path + "*"), class_num)
        else:
            class_list = glob.glob(self.imgs_path + "*")
            
        self.data = []
        classes_names = []
        for class_path in class_list:
            class_name = class_path.split("/")[-1]
            
            class_images = glob.glob(class_path + "/*.jpeg")
            if train:
                if len(class_images) < num_img:
                    continue
                else:
                    classes_names.append(class_name)
                    class_images = random.sample(class_images, num_img)
                    for img_path in class_images:
                        self.data.append([img_path, class_name])
            else:
                if len(class_images) < num_img:
                    classes_names.append(class_name)
                    for img_path in class_images:
                        self.data.append([img_path, class_name])
                else:
                    continue
        self.img_dim = (img_size, img_size)
        self.classes = classes_names
        self.class_d = {}
        for i, cl in enumerate(self.classes):
            self.class_d[cl] = i


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)

        class_id = self.class_d[class_name]
        preprocess = transforms.Compose([
            transforms.Resize(self.img_dim),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        try:
            img_tensor = preprocess(img)
        except RuntimeError as e:
            print(str(e))
            print(img_path)
            pass
        return img_tensor, class_id
