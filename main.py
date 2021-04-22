#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import torch
from model import FNM_model as FNM_model
from train_single_fnm import train_single_fnm
from Generate_Image import Generate_Image
from util.Load_PretrainModel import Load_PretrainModel
from Pretrained.VGGFace2.resnet50_ft_dims_2048 import resnet50_ft
from util.args_warning import args_warning, evl_args_warning

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameters
    parser.add_argument('-lambda_l1', type=float, default=0.001, help='weight of the loss for L1 texture loss [default: 0.001]')
    parser.add_argument('-lambda_fea', type=float, default=1000, help='weight of the loss for face model feature loss [default: 1000]')
    parser.add_argument('-lambda_reg', type=float, default=1e-5, help='weight of the loss for L2 regularitaion loss [default: 1e-5]')
    parser.add_argument('-lambda_gan', type=float, default=1, help='weight of the loss for gan loss [default: 1]')
    parser.add_argument('-lmbda_gp', type=float, default=10, help='Gradient Penalty Coeficient [default: 10]')

    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate [default: 0.0001]')
    parser.add_argument('-beta1', type=float, default=0, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.9, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 8]')
    parser.add_argument('-snapshot-dir', type=str, default='snapshot', help='where to save the snapshot while training')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

    # data source
    parser.add_argument('-data-place', type=str, default=None, help='prepared data path to run program')
    parser.add_argument('-front-list', type=str, default=None, help='csv file to load normal set for training')
    parser.add_argument('-profile-list', type=str, default=None, help='csv file to load source set for validation')
    parser.add_argument('-gen-list', type=str, default=None, help='csv file to load source set for evaluation')
    parser.add_argument('-Channel', type=int, default=3, help='initial Number of Channel [default: 3 (RGB Three Channel)]')
    parser.add_argument('-num-critic-G', default=1, type=int, help='number of iterations of changing the training between G and D')
    parser.add_argument('-num-critic-D', default=1, type=int, help='number of iterations of changing the training between G and D')
    # model
    parser.add_argument('-VGGFace2', action='store_true', default=True, help='enable the VGGFace2 encoding model')
    parser.add_argument('-ArcFace', action='store_true', default=False, help='enable the ArcFace encoding model')
    # test
    parser.add_argument('-model-select', type=str, default='Light_CNN_29', help='Model Select')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{date}/{epoch}) [default: None]')
    parser.add_argument('-generate', action='store_true', default=None, help='Generate normalized image from given image')
    parser.add_argument('-generate-place', type=str, default='GenerateImage', help='place to save the generated images while testing')
    parser.add_argument('-encoder', action='store_true', default=False, help='Extract identity features by idependent model')
    parser.add_argument('-decoder', action='store_true', default=True, help='Generate the normalized image')
    # option
    parser.add_argument('-resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-pretrain', default='', type=str, metavar='PATH', help='path to the pretrain model (default:none)')


    args = parser.parse_args()

    # Load the Expert Network
    C = resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
    print('VGGFace2 model built successfully')
    if (args.generate):
        evl_args_warning(args)
        print('\nLoading model from [%s]...' % args.snapshot)
        checkpoint = torch.load('{}_checkpoint.pth.tar'.format(args.snapshot))
        D = FNM_model.Discriminator(args)
        G = FNM_model.Generator(args)
        D.load_state_dict(checkpoint['D_model'])
        G.load_state_dict(checkpoint['G_model'])
        Generate_Image(D, G, C, args)
    else:
        args.snapshot_dir = os.path.join(args.snapshot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(args.snapshot_dir)
        args_warning(args)
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                D = FNM_model.Discriminator(args)
                G = FNM_model.Generator(args)
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                D.load_state_dict(checkpoint['D_model'])
                G.load_state_dict(checkpoint['G_model'])
                print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
                print("=> loaded Discriminator Pretrained Model")
                print("=> loaded Generator Pretrained Model")
                train_single_fnm(D, G, C, args)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        elif args.pretrain:
            D_Pretrain_Path = args.pretrain + '_D.pt'
            G_Pretrain_Path = args.pretrain + '_G.pt'
            if os.path.isfile(D_Pretrain_Path) and os.path.isfile(G_Pretrain_Path):
                print("=> loading Discriminator Pretrain Model '{}'".format(D_Pretrain_Path))
                print("=> loading Generator Pretrain Model '{}'".format(G_Pretrain_Path))
                D = FNM_model.Discriminator(args)
                G = FNM_model.Generator(args)
                D_Pretrain_dict = torch.load(D_Pretrain_Path)
                G_Pretrain_dict = torch.load(G_Pretrain_Path)
                D_model_dict = D.state_dict()
                G_model_dict = G.state_dict()
                D_MODEL = Load_PretrainModel(D, D_model_dict, D_Pretrain_dict)
                G_MODEL = Load_PretrainModel(G, G_model_dict, G_Pretrain_dict)
                train_single_fnm(D_MODEL, G_MODEL, C, args)
            else:
                print("=> no Pretrain Model found at '{}'".format(args.pretrain))
        else:
            D = FNM_model.Discriminator(args)
            G = FNM_model.Generator(args)
            train_single_fnm(D, G, C, args)





