#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.autograd import Variable
from util.DataAugmentation import FaceIdPoseDataset
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from util.ConcatPath import ConcatPath




def Generate_Image(D_model, G_model, C_model, args):

    if args.decoder:
        save_dir = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Image')
    elif args.encoder:
        save_dir = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Feature')

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.cuda:
        D_model.to(device)
        G_model.to(device)
        C_model.to(device)

    G_model.eval()
    D_model.eval()
    C_model.eval()

    count = 0
    # Load augmented data
    transformed_dataset = FaceIdPoseDataset(args.gen_list, args.data_place,
                                            transform=transforms.Compose([torchvision.transforms.Resize(224),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_image = batch_data[0].to(device)
            minibatch_size = len(batch_image)

            _, FeaMap = C_model(batch_image)
            generated = G_model(FeaMap)
            batchImageName = batch_data[1]

            if args.decoder:
                for j, imgName in enumerate(batchImageName):
                    save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                    save_generated_image = np.squeeze(save_generated_image)
                    save_generated_image = (save_generated_image+1)/2.0 * 255.
                    save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB
                    folder_split = imgName.split('/')
                    filename, Save_Folder = ConcatPath(save_dir, folder_split, '.jpg')
                    if not os.path.isdir(Save_Folder):
                        os.makedirs(Save_Folder)
                    print('saving {}'.format(filename))
                    misc.imsave(filename, save_generated_image.astype(np.uint8))

            if args.encoder:
                features = (G_model.features)
                features = (features.data).cpu().numpy()
                SaveFeature(features, batchImageName, save_dir)
            count += minibatch_size
            print("Finish Processing {} images...".format(count))
