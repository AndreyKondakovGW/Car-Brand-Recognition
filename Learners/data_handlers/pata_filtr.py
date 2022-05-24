import glob
from PIL import Image
import numpy as np
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def filter_data(path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + path
    print(path)
    class_list = glob.glob(path + "*")
    
    for cl in class_list:
        class_images = glob.glob(cl + "/*.jpeg")
        for img_path in class_images:
            img = Image.open(img_path)
            a = np.array(img)
            if len(a.shape) != 3:
                print(img_path)
                print("Shape " + str(a.shape))
            
    
filter_data("/../../data/images_balance/")