#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import torch
from model import FNM_model as FNM_model
from train_single_DRGAN import train_single_DRGAN
from Generate_Image import Generate_Image
from util.Load_PretrainModel import Load_PretrainModel
from model.model_irse import IR_50
from Pretrained.VGGFace2.resnet50_ft_dims_2048 import resnet50_ft

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameters
    parser.add_argument('-lambda_l1', type=float, default=0.001, help='weight of the loss for L1 texture loss [default: 0.001]')
    parser.add_argument('-lambda_fea', type=float, default=1000, help='weight of the loss for face model feature loss [default: 1000]')
    parser.add_argument('-lambda_reg', type=float, default=1e-5, help='weight of the loss for L2 regularitaion loss [default: 1e-5]')
    parser.add_argument('-lambda_gan', type=float, default=1, help='weight of the loss for gan loss [default: 1]')
    parser.add_argument('-lmbda_gp', type=float, default=10, help='Gradient Penalty Coeficient [default: 10]')

    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-beta1', type=float, default=0, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.9, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-snapshot-dir', type=str, default='snapshot', help='where to save the snapshot while training')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('-start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

    # data souce
    parser.add_argument('-data-place', type=str, default=None, help='prepared data path to run program')
    parser.add_argument('-front-list', type=str, default=None, help='csv file to load image for training')
    parser.add_argument('-profile-list', type=str, default=None, help='csv file to load image for validation')
    parser.add_argument('-val-list', type=str, default=None, help='csv file to load image for validation')
    parser.add_argument('-Channel', type=int, default=3, help='initial Number of Channel [default: 3 (RGB Three Channel)]')
    parser.add_argument('-num-critic-G', default=1, type=int, help='number of iterations of changing the training between C, G and D')
    parser.add_argument('-num-critic-D', default=1, type=int, help='number of iterations of changing the training between C, G and D')
    # model
    parser.add_argument('-ArcFace', action='store_true', default=False, help='enable the encoding model')
    parser.add_argument('-VGGFace2', action='store_true', default=False, help='enable the encoding model')
    # option
    parser.add_argument('-step-learning', action='store_true', default=False, help='enable lr step learning')
    parser.add_argument('-lr-decay', type=float, default=0.1, help='initial decay learning rate [default: 0.1]')
    parser.add_argument('-lr-step', type=int, default=35, help='Set Step to change lr by multiply lr-decay thru every lr-step epoch [default: 35]')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-test', action='store_true', default=None, help='Test Network Performance')
    parser.add_argument('-generate-place', type=str, default='GenerateImage', help='where to save the generated images while testing')
    parser.add_argument('-resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-pretrain', default='', type=str, metavar='PATH', help='path to the pretrain model (default:none)')


    args = parser.parse_args()

    if (args.generate):
        if args.encoder is False and args.decoder is False:
            print("Sorry, please set encoder or decoder to true while trigger generate feature")
            exit()
        elif args.data_place is None:
            print("Sorry, please set data place for your input dat")
            exit()
        else:
            if args.snapshot is None:
                print("Sorry, please set snapshot path while generate")
                exit()
            else:
                print('\nLoading model from [%s]...' % args.snapshot)
                checkpoint = torch.load('{}_checkpoint.pth.tar'.format(args.snapshot))
                if not args.multi_DRGAN:
                    D = single_model.Discriminator(args)
                    G = single_model.Generator(args)
                    if args.OTS_Cla:
                        C = IR_50(args, 112)
                    else:
                        C = single_model.Classifier(args)
                else:
                    if args.batch_size % args.Image_IDNum == 0:
                        D = multi_model.Discriminator(args)
                        G = multi_model.Generator(args)
                    else:
                        print("Please give valid combination of batch_size, images_IDNum")
                        exit()

                D.load_state_dict(checkpoint['D_model'])
                G.load_state_dict(checkpoint['G_model'])
                C.load_state_dict(checkpoint['C_model'])
                Generate_Image(D, G, C, args)
    else:
        args.snapshot_dir = os.path.join(args.snapshot_dir, 'Single', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(args.snapshot_dir)

        print("Parameters:")
        for attr, value in sorted(args.__dict__.items()):
            text = "\t{}={}\n".format(attr.upper(), value)
            print(text)
            with open('{}/Parameters.txt'.format(args.snapshot_dir), 'a') as f:
                f.write(text)

        if args.front_list is None or args.profile_list is None or args.val_list is None:
            print("Sorry, please set csv-file for your front/profile/validation data")
            exit()

        if args.data_place is None:
            print("Sorry, please set -data-place for your input data")
            exit()

        if not args.ArcFace and not args.VGGFace2:
            print("Sorry, please select valid option")
            # We can make one of them as the default setting (when both is fault)
            exit()

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                D = single_model.Discriminator(args)
                G = single_model.Generator(args)
                if args.OTS_Cla:
                    C = IR_50(args, 112)
                else:
                    C = single_model.Classifier(args)
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                D.load_state_dict(checkpoint['D_model'])
                G.load_state_dict(checkpoint['G_model'])
                C.load_state_dict(checkpoint['C_model'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                print("=> loaded Discriminator Pretrained Model")
                print("=> loaded Generator Pretrained Model")
                train_single_DRGAN(D, G, C, args)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        elif args.pretrain:
            D_Pretrain_Path = args.pretrain + '_D.pt'
            G_Pretrain_Path = args.pretrain + '_G.pt'
            C_Pretrain_Path = args.pretrain + '_C.pt'
            if os.path.isfile(D_Pretrain_Path) and os.path.isfile(G_Pretrain_Path) and os.path.isfile(C_Pretrain_Path):
                print("=> loading Discriminator Pretrain Model '{}'".format(D_Pretrain_Path))
                print("=> loading Cenerator Pretrain Model '{}'".format(G_Pretrain_Path))
                print("=> loading Cenerator Pretrain Model '{}'".format(C_Pretrain_Path))
                if args.OTS_Cla:
                    C = IR_50(args, 112)
                else:
                    C = single_model.Classifier(args)
                D = single_model.Discriminator(args)
                G = single_model.Generator(args)
                D_Pretrain_dict = torch.load(D_Pretrain_Path)
                G_Pretrain_dict = torch.load(G_Pretrain_Path)
                C_Pretrain_dict = torch.load(C_Pretrain_Path)
                D_model_dict = D.state_dict()
                G_model_dict = G.state_dict()
                C_model_dict = C.state_dict()
                D_MODEL = Load_PretrainModel(D, D_model_dict, D_Pretrain_dict)
                G_MODEL = Load_PretrainModel(G, G_model_dict, G_Pretrain_dict)
                C_MODEL = Load_PretrainModel(C, C_model_dict, C_Pretrain_dict)
                train_single_DRGAN(D_MODEL, G_MODEL, C_MODEL, args)
            else:
                print("=> no Pretrain Model found at '{}'".format(args.pretrain))
        else:

            if args.ArcFace:
                OTS_C = IR_50(args, 112)
                OTS_C_Pretrain_dict = torch.load('Pretrained/IR-50_MS1M_ArcFace/backbone_ir50_ms1m_epoch63.pth')
                OTS_C_model_dict = OTS_C.state_dict()
                C = Load_PretrainModel(OTS_C, OTS_C_model_dict, OTS_C_Pretrain_dict)
                print('ArcFace model built successfully')
            elif args.VGGFace2:
                C = resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
                print('VGGFace2 model built successfully')

            D = FNM_model.Discriminator(args)
            G = FNM_model.Generator(args)
            train_single_DRGAN(D, G, C, args)





