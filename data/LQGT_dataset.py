import os
import numpy as np
import cv2
import random

from torch.utils.data import Dataset


class LQGTDataset(Dataset):
    def __init__(self, db_path, transform):
        super(LQGTDataset, self).__init__()
        self.db_path = db_path
        self.transform = transform

        self.dirname_GT = os.path.join(self.db_path, 'track1_DF2K/Corrupted-tr-y')
        self.dirname_LQ = os.path.join(self.db_path, 'track1_DF2K/Corrupted-tr-x')
    
        self.filelist_GT = os.listdir(self.dirname_GT)
        self.filelist_LQ = os.listdir(self.dirname_LQ)
    
    def __len__(self):
        return len(self.filelist_GT)

    def __getitem__(self,idx):
        data_ratio = len(self.filelist_LQ) / len(self.filelist_GT)

        img_name_GT = self.filelist_GT[idx]
        img_GT = cv2.imread(os.path.join(self.dirname_GT, img_name_GT), cv2.IMREAD_COLOR)
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
        img_GT = np.array(img_GT).astype('float32') / 255

        LQ_idx = int(idx * (data_ratio + random.randint(1, 5)) + random.randint(1,10)) % len(self.filelist_LQ)
        img_name_LQ = self.filelist_LQ[LQ_idx]
        img_LQ = cv2.imread(os.path.join(self.dirname_LQ, img_name_LQ))
        img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_BGR2RGB)
        img_LQ = np.array(img_LQ).astype('float32') / 255

        # img_LQ: H x W x C (numpy array)
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}

        if self.transform:
            sample = self.transform(sample)

        return sample
