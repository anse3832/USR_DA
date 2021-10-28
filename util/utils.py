import torch
import torchvision

import math
import cv2
import numpy as np
from scipy.ndimage import rotate


class RandCrop(object):
    def __init__(self, crop_size, scale):
        # if output size is tuple -> (height, width)
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        
        self.scale = scale

    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        h, w, c = img_LQ.shape
        new_h, new_w = self.crop_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img_LQ_crop = img_LQ[top: top+new_h, left: left+new_w, :]

        h, w, c = img_GT.shape
        top = np.random.randint(0, h - self.scale*new_h)
        left = np.random.randint(0, w - self.scale*new_w)
        img_GT_crop = img_GT[top: top + self.scale*new_h, left: left + self.scale*new_w, :]

        sample = {'img_LQ': img_LQ_crop, 'img_GT': img_GT_crop}
        return sample


class RandRotate(object):
    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        prob_rotate = np.random.random()
        if prob_rotate < 0.25:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        elif prob_rotate < 0.5:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        elif prob_rotate < 0.75:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample


class RandHorizontalFlip(object):
    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        prob_lr = np.random.random()
        if prob_lr < 0.5:
            img_LQ = np.fliplr(img_LQ).copy()
            img_GT = np.fliplr(img_GT).copy()
        
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # img_LQ : H x W x C (numpy array) -> C x H x W (torch tensor)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        img_LQ = img_LQ.transpose((2, 0, 1))
        img_GT = img_GT.transpose((2, 0, 1))

        img_LQ = torch.from_numpy(img_LQ)
        img_GT = torch.from_numpy(img_GT)

        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample


class VGG19PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layer=35):
        super(VGG19PerceptualLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters
        for name, param in self.features.named_parameters():
            param.requires_grad = False
    
    def forward(self, source, target):
        vgg_loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return vgg_loss
        

