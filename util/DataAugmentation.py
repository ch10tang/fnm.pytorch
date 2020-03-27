#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]
from __future__ import print_function, division
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FaceIdPoseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.imgFrame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.imgFrame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgFrame.ix[idx, 0])
        imgName = self.imgFrame.ix[idx, 0]
        if not os.path.isfile(img_path):
            print('>>> No Such File: {}'.format(img_path))
            exit()

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return [image, imgName]



