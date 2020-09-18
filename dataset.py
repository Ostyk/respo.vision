from torch import nn
import torch
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from imgaug import augmenters as iaa
import imgaug as ia

import cv2
from ipywidgets import IntProgress
from tqdm import tqdm
from PIL import Image

mapper = {"neg": 0, "pos": 1}

class FootballFrameDataset(object):
    
    def __init__(self, df, transforms = None, normalize = False, size=(299, 299)):
                 
        self.network_size = size
        self.normalize = normalize
        self.transforms = transforms
        
        self.imgs = []
        self.labels = []
        for num, info in df.iterrows():
            img_path = os.path.join(info.root, info.img_name)
            self.imgs.append(img_path)
            self.labels.append(info.frame)
            
    def __getitem__(self, idx):
                    
        img = Image.open(self.imgs[idx])
        img = img.resize(self.network_size, Image.ANTIALIAS)
            
        target = mapper[self.labels[idx]]
            
        if self.transforms is not None:
            #img = Image.fromarray(np.uint8(img*255))
            img = self.transforms(img)
            
        img = np.array(img)
        img = np.moveaxis(img, -1, -0).astype(np.float32)
        if self.normalize:
            img  *= (1. / 255)
        return img, target
    
    def __len__(self):
        return len(self.imgs)