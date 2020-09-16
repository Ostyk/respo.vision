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

class LeafDatasetPytorch(object):
    """
    Class for images based on data_create notebook
    
    """
    
    def __init__(self, root, subset, transforms, n_stack, normalize=False, n_classes=2):
        """
        :param root: dir path
        :param subset: train val tes
        :param transforms: data augmentation
        :param n_stack: equal to pyramid n_stack
        """
        
        self.transforms = transforms
        self.subset = subset
        self.n_stack = n_stack
        self.normalize = normalize
        self.n_classes = n_classes
        mask_path = os.path.join(root, 'polygons.json')
        with open(mask_path) as f:
            self.polygons = json.load(f)

        self.root = os.path.join(root, self.subset)
        self.imgs = sorted([i for i in os.listdir(self.root) if i.endswith('.png')])

    def __getitem__(self, idx):
        
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR) 
        if self.normalize:
            img = img * (1. / 255)

        polygons = self.polygons[self.imgs[idx]]['polygons']
        # to do fix the data aug

        #convert to give ground truths at differnt resolutions
        mask_2d = self.Polygon_fill_and_contour(img, polygons)
        
        if self.transforms is not None:
            img, mask_2d = self.transforms(img, mask_2d)
        
        labels = torch.unique(torch.tensor(mask_2d))  #pixel wise 0-Background, 1-edge, 2-interior
        
        mask_dict_resolutions = self.resolution_ground_truth(mask_2d, self.n_stack, self.n_classes)
        
        target = {}
        target["labels"] = labels
        target["masks"] = mask_dict_resolutions
        target["image_id"] = torch.tensor([idx])
        target["num_polygons"] = len(polygons)
        
        #switching for pytorch channels first
        img = np.moveaxis(img, -1, 0).astype(np.float32)

        
#         if self.transforms is not None:
#             img, target["masks"] = self.transforms(img, target["masks"])

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def _MutipleMasksFromPolygons(self, img, polygons):
        """
        creates 3D array of masks
        returns mask-r-cnn style masks
        """
        mask = np.zeros([len(polygons), img.shape[1], img.shape[2]], dtype=np.uint8)
        for index, coords in tqdm(enumerate(polygons), total=len(polygons), desc='polygons'):
            cv2.fillConvexPoly(mask[index, :, :], np.array(coords, 'int32'), color=1)
        print(mask.shape)
        return mask
         
    @staticmethod
    def Polygon_fill_and_contour(img, polygons):
        """
        Returns 2D array of masks. Filled with gray, edge coloured in white.
        """
        edge_color = 2 #white
        interior_color = 1 #gray
        contours = [np.vstack([np.array(i) for i in j]) for j in polygons]
        mask_test = np.zeros([img.shape[0], img.shape[1]], dtype=np.int32)
        for curr_pol in polygons:
            cv2.fillConvexPoly(mask_test, np.array(curr_pol, 'int32'), color=interior_color)
        cv2.drawContours(mask_test,  contours, contourIdx=-1, color=edge_color, thickness=4)
        return mask_test
    
    @staticmethod
    def resolution_ground_truth(mask, n_stack, n_classes):
        """
        retrieves ground truth masks at different resolutions in a consecutive manner
        :param mask: input mask
        :param n_stack: how many resolutions --> n_stack should be equal to the pyramid size
        :return: dict of n_stack number as key and 2D numpy array as the downsampled masks
        """
        def downsample_ground_truth(mask_2d):
            """ applies maxpooling for the ground truth mask"""
            return torch.nn.functional.max_pool2d(mask_2d,
                                                  kernel_size=(2,2),
                                                  stride=2,
                                                  padding=0
                                                 )

        main = {}
        mask = torch.Tensor(np.expand_dims(mask, 0)) #for torch tensor purposes
        if n_classes==3:
            main.update({0:mask.type(torch.long)}) # original input
        else:
            obj_ids = torch.unique(mask)[1:] # n_classes
            masks = mask == obj_ids[:, None, None]
            main.update({0:masks.type(torch.long)}) # original input
        for n in range(1, n_stack):
            mask = downsample_ground_truth(mask)
            if n_classes==3:
                main.update({n:mask.type(torch.long)})
            else:
                obj_ids = torch.unique(mask)#[1:]
                masks = mask == obj_ids[:, None, None]
                main.update({n:masks.type(torch.long)})

        return main
    