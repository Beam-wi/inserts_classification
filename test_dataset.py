import torch
import torchvision
import numpy as np
import cv2
import math
import yaml

from dataset.dataset import Dataset


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    print(cfg)

    dataset = Dataset(cfg['train_data_dir'], cfg['image_size'], cfg['crop_ratio'], phase='train')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()

        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        img *= cfg['imagenet_default_std']
        img += cfg['imagenet_default_mean']
        img *= 255.0
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite('dataset_test_img.jpg', img)