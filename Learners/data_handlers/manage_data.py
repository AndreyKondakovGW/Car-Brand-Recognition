import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil

def move_files(images, src, dst):
    for img in images:
        shutil.copy2("{}/{}".format(src, img), dst+"/")
    
def move_images_from_table(cv_path, output_dir, img_folder):
    """
        Load images from image column in csv table
    """
    df = pd.read_csv(cv_path)

    vc = df['label'].value_counts()
    classes = vc.index

    if (not os.path.isdir(output_dir)):
        print(f"Dir {output_dir} is not exist")
        return
    images_num = 0
    for cl in tqdm(classes):
        print('Download class '+ str(cl))
        class_dir = "{}/{}".format(output_dir, cl)
        os.system("mkdir " + class_dir)
        images = df[df['label'] == cl]['image']
        class_folder = img_folder + "/" + str(cl)
        move_files(images, class_folder,  class_dir)
        images_num += len(images)
    print("Successfully moved data {} images".format(images_num))
    os.system('du -h {}'.format(output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IamgeDownloader From Server')
    parser.add_argument('--csv_name', type=str, default='./data/class_exaples.csv', help='')
    parser.add_argument('--img_folder', default='./data/images_balance',  type=str, help='')
    parser.add_argument('--output_folder', default='./data/examples/examples_all', type=str, help='')
    args = parser.parse_args()

    move_images_from_table(
        cv_path = args.csv_name,
        output_dir = args.output_folder,
        img_folder = args.img_folder
    )
    print('Run')
