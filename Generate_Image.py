#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.autograd import Variable
from util.DataAugmentation import FaceIdPoseDataset
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader
from torchvision import transforms
from util.ConcatPath import ConcatPath



def Generate_Image(D_model, G_model, C_model, args):

    Ny = args.Ny
    Nz = args.Nz


    if args.decoder:
        save_dir = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Image')
    elif args.encoder:
        save_dir = '{}/{}_generated/{}'.format(args.generate_place, args.snapshot, 'Feature')

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    if args.cuda:
        D_model.cuda()
        G_model.cuda()
        C_model.cuda()

    G_model.eval()
    D_model.eval()
    C_model.eval()

    count = 0
    # Load augmented data
    transformed_dataset = FaceIdPoseDataset(args.gen_csv_file, args.data_place,
                                            transform = transforms.Compose([Resize((110,110)), RandomCrop((96,96))]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)

    if not(args.multi_DRGAN):

        for i, batch_data in enumerate(dataloader):
            batch_image = torch.FloatTensor(batch_data[0].float())
            minibatch_size = len(batch_image)
            if args.specify_code is None:
                batch_yaw_label = torch.LongTensor(np.ones(minibatch_size) * 6)#batch_data[3]
                batch_yaw_code = torch.FloatTensor(one_hot(batch_yaw_label, Ny))
            else:
                tmp = args.specify_code
                yaw_code = torch.LongTensor(np.ones(minibatch_size) * int(tmp))
                batch_yaw_code = torch.FloatTensor(one_hot(yaw_code, Ny))

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_yaw_code = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_yaw_code.cuda()

            with torch.no_grad():
                batch_image, fixed_noise, batch_yaw_code = \
                    Variable(batch_image), Variable(fixed_noise), Variable(batch_yaw_code)

            generated = G_model(batch_image, batch_yaw_code, fixed_noise)
            FC_Out = C_model(batch_image)

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
                    # filename = os.path.join(Save_Folder, '{}'.format(folder_split[2]))
                    print('saving {}'.format(filename))
                    misc.imsave(filename, save_generated_image.astype(np.uint8))

            if args.encoder:
                features = (G_model.features)
                # features = (D_model.features)
                features = (features.data).cpu().numpy()
                SaveFeature(features, batchImageName, save_dir)
            count += minibatch_size
            print("Finish Processing {} images...".format(count))

    else:
        for i, batch_data in enumerate(dataloader):
            batch_image = torch.FloatTensor(batch_data[0].float())
            minibatch_size = len(batch_image)
            minibatch_size_unique = len(batch_image) // args.Image_IDNum

            if args.specify_code is None:
                batch_yaw_label = torch.LongTensor(np.ones(minibatch_size) * 6)
                batch_yaw_code = torch.FloatTensor(one_hot(batch_yaw_label, Ny))
            else:
                tmp = args.specify_code
                yaw_code = torch.LongTensor(np.ones(minibatch_size) * int(tmp))
                batch_yaw_code = torch.FloatTensor(one_hot(yaw_code, Ny))

            if args.specify_code is None:
                batch_yaw_label_unique = torch.LongTensor(np.ones(minibatch_size_unique) * 6)
                batch_yaw_code_unique = torch.FloatTensor(one_hot(batch_yaw_label_unique, Ny))
            else:
                tmp = args.specify_code
                yaw_code_unique = torch.LongTensor(np.ones(minibatch_size_unique) * int(tmp))
                batch_yaw_code_unique = torch.FloatTensor(one_hot(yaw_code_unique, Ny))

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))
            fixed_noise_unique = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size_unique, Nz)))

            if args.cuda:
                batch_image, fixed_noise, fixed_noise_unique, batch_yaw_code, batch_yaw_code_unique= \
                    batch_image.cuda(), fixed_noise.cuda(), fixed_noise_unique.cuda(), batch_yaw_code.cuda(), batch_yaw_code_unique.cuda()

            batch_image, fixed_noise, fixed_noise_unique, batch_yaw_code, batch_yaw_code_unique = \
                Variable(batch_image), Variable(fixed_noise), Variable(fixed_noise_unique), Variable(batch_yaw_code), Variable(batch_yaw_code_unique)

            # Generate Image
            generated = G_model(batch_image, batch_yaw_code, fixed_noise, single=True)
            generated_Unique = G_model(batch_image, batch_yaw_code_unique, fixed_noise_unique)

            batchImageName = batch_data[1]
            batchImageName_Unique = batch_data[1][::args.Image_IDNum]

            if args.decoder:
                for j, imgName in enumerate(batchImageName):
                    tmp = imgName.split('/')
                    save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                    save_generated_image = np.squeeze(save_generated_image)
                    save_generated_image = (save_generated_image+1)/2.0 * 255.
                    save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB
                    fileName = '_'.join(tmp)

                    if args.specify_code != None:
                        FilePath = os.path.join(save_dir, '{}_{}.jpg'.format(fileName[:-4], args.specify_code))
                    else:
                        FilePath = os.path.join(save_dir, '{}.jpg'.format(fileName[:-4]))

                    print('saving {}'.format(FilePath))
                    misc.imsave(FilePath, save_generated_image.astype(np.uint8))

                for j, imgName in enumerate(batchImageName_Unique):
                    tmp = imgName.split('/')
                    save_generated_image = generated_Unique[j].cpu().data.numpy().transpose(1, 2, 0)
                    save_generated_image = np.squeeze(save_generated_image)
                    save_generated_image = (save_generated_image + 1) / 2.0 * 255.
                    save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB

                    fileName = '_'.join(tmp)

                    if args.specify_code != None:
                        FilePath = os.path.join(save_dir, 'Multi_{}_{}.jpg'.format(fileName[:-4], args.specify_code))
                    else:
                        FilePath = os.path.join(save_dir, 'Multi_{}.jpg'.format(fileName[:-4]))

                    print('saving {}'.format(FilePath))
                    misc.imsave(FilePath, save_generated_image.astype(np.uint8))

            if args.encoder:
                features = (G_model.features)
                #features = D_model(batch_image)
                features = (features.data).cpu().numpy()
                SaveFeature(features, batchImageName, save_dir, args)
            count += minibatch_size
            print("Finish Processing {} images...".format(count))