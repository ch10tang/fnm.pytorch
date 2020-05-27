#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import torch
from util.DataAugmentation import FaceIdPoseDataset
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from util.ConcatPath import ConcatPath
from util.Load_OTS_Pretrained import Load_OTS_Pretrained
from Pretrained.VGGFace2.resnet50_ft_dims_2048 import resnet50_ft
from model.model_irse import IR_50
from model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from util.InputSize_Select import InputSize_Select
import cv2



def Generate_Image(D_model, G_model, C_model, args):


    BACKBONE_DICT = {'IR-50': IR_50(112),
                         'Light_CNN_9': LightCNN_9Layers(),
                         'Light_CNN_29': LightCNN_29Layers(),
                         'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                         'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
                         }
    BACKBONE = BACKBONE_DICT[args.model_select]
    Extractor = Load_OTS_Pretrained(BACKBONE, args)


    if args.decoder:
        save_dir_img = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Image')
        if not os.path.isdir(save_dir_img): os.makedirs(save_dir_img)
    if args.encoder:
        save_dir_fea = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Feature')
        if not os.path.isdir(save_dir_fea): os.makedirs(save_dir_fea)

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.cuda:
        D_model.to(device)
        G_model.to(device)
        C_model.to(device)
        Extractor.to(device)

    G_model.eval()
    D_model.eval()
    C_model.eval()
    Extractor.eval()

    count = 0
    # Load augmented data
    transformed_dataset = FaceIdPoseDataset(args.gen_list, args.data_place,
                                            transform=transforms.Compose([torchvision.transforms.Resize(224),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_image = ((batch_data[0] + 1) * 127.5).to(device)
            minibatch_size = len(batch_image)

            _, FeaMap = C_model(batch_image)
            generated = G_model(FeaMap)
            batchImageName = batch_data[1]

            if args.decoder:
                for j, imgName in enumerate(batchImageName):
                    save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                    save_generated_image = np.squeeze(save_generated_image)
                    save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB
                    folder_split = imgName.split('/')
                    filename, Save_Folder = ConcatPath(save_dir_img, folder_split, '.jpg')
                    if not os.path.isdir(Save_Folder): os.makedirs(Save_Folder)
                    cv2.imwrite(filename, save_generated_image.astype(np.uint8))

            if args.encoder:
                generated = InputSize_Select(generated, args)
                _ = Extractor(generated)
                try: features = Extractor.feature
                except: features = Extractor.module.feature
                features = (features.data).cpu().numpy()
                SaveFeature(features, batchImageName, save_dir_fea)
            count += minibatch_size
            print("Finish Processing {} images...".format(count))
