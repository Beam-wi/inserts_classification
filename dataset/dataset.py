import os
import sys
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
import numpy as np
import random
import cv2
from PIL import Image
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import letterbox_image_pil, letterbox_image_cv


class Dataset(data.Dataset):

    def __init__(self, data_dir, input_size, crop_ratio, phase='train'):
        self.data_dir = data_dir
        self.phase = phase
        self.input_size = np.array(input_size)
        self.crop_ratio = crop_ratio
        self.enlarge_size = (self.input_size / self.crop_ratio).astype(np.int)

        self.cls_names = os.listdir(self.data_dir)

        self.category_id_to_name = {k: v for k, v in enumerate(self.cls_names)}
        self.category_name_to_id = {v: k for k, v in self.category_id_to_name.items()}

        self.data_list = make_data_list_from_dir(self.data_dir, self.category_name_to_id)

        print(self.cls_names)
        print(f'input size:{self.input_size}, enlarg size:{self.enlarge_size}')



        self.aug = A.Compose([A.ColorJitter(),
                              A.Rotate(border_mode=0, value=(128,128,128)),
                              A.Flip(),
                              A.RandomCrop(self.input_size[0], self.input_size[1]),
                              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                              ToTensorV2(),
        ])

        self.infer_transfom = A.Compose([A.CenterCrop(self.input_size[0], self.input_size[1]),
                              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                              ToTensorV2(),
        ])


    def __getitem__(self, index):
        sample = self.data_list[index]
        img_path = sample[0]
        cls_idx = sample[1]

        img = cv2.imread(img_path)
        if img is None:
            print("Error: read %s fail" % img_path)
            exit()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[0:2]

        img, _, _ = letterbox_image_cv(img, self.enlarge_size)

        if self.phase == 'train':
            img_tensor = self.aug(image=img)['image']
        else:
            img_tensor = self.infer_transfom(image=img)['image']

        return img_tensor, cls_idx

    def __len__(self):
        return len(self.data_list)


def make_data_list_from_dir(data_dir, category_name_to_id):
    class_names = os.listdir(data_dir)
    data_list = []
    for cls_name in class_names:
        sub_dir = os.path.join(data_dir, cls_name)
        files = os.listdir(sub_dir)
        for each_file in files:
            img_path = os.path.join(sub_dir, each_file)
            cls_id = category_name_to_id[cls_name]
            data_list.append([img_path, cls_id])
    return data_list
    
    
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight               
