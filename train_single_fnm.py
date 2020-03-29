#/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import torchvision
from util.DataAugmentation import FaceIdPoseDataset
from util.checkpoint import save_checkpoint
from util.Gradient_Control import enable_gradients, disable_gradients
import torch.backends.cudnn as CUDNN
from util.Validation_CFP import Validation_CFP
from util.Validation_CFP_Single import Validation_CFP_Single
import torch.nn.functional as F
from util.log_learning import log_learning
import torchvision.utils as vutils

def train_single_DRGAN(D_model, G_model, C_model, args):

    writer = SummaryWriter()

    D_lr = args.lr
    G_lr = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if args.cuda:
        D_model.to(device)
        G_model.to(device)
        C_model.to(device)

    optimizer_D = optim.Adam(D_model.parameters(), lr=D_lr, betas=(beta1_Adam, beta2_Adam), weight_decay=args.lambda_reg)
    optimizer_G = optim.Adam(G_model.parameters(), lr=G_lr, betas=(beta1_Adam, beta2_Adam), weight_decay=args.lambda_reg)


    if args.resume:
        checkpoint = torch.load(args.resume)
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])


    steps = 0
    CUDNN.benchmark = True

    for epoch in range(args.start_epoch, args.epochs+1):

        D_model.train()
        G_model.train()
        C_model.eval()

        # Load augmented data
        profile_dataset = FaceIdPoseDataset(args.profile_list, args.data_place,
                                                transform=transforms.Compose([torchvision.transforms.Resize(250),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        front_dataset = FaceIdPoseDataset(args.front_list, args.data_place,
                                                transform=transforms.Compose([torchvision.transforms.Resize(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        profile_dataloader = DataLoader(profile_dataset, batch_size=args.batch_size, shuffle=True)#, num_workers=6)
        front_dataloader = DataLoader(front_dataset, batch_size=args.batch_size, shuffle=True)  # , num_workers=6)

        for idx, _ in enumerate(profile_dataloader):

            batch_profile, imageName_profile = next(iter(profile_dataloader))
            batch_front, imageName_front = next(iter(front_dataloader))

            batch_profile = ((batch_profile + 1) * 127.5).to(device)
            batch_front = ((batch_front + 1) * 127.5).to(device)
            steps += 1

            enable_gradients(D_model)
            disable_gradients(C_model)
            disable_gradients(G_model)

            if steps < 25 and epoch == 1:
                critic = 25
            else:
                critic = args.num_critic_D
            for _ in range(0, critic):
                D_model.zero_grad()
                # Create Encoder Feature Map / Get the Real images' Features

                _, Front_FeaMap = C_model(batch_front)
                _, Profile_FeaMap = C_model(batch_profile)
                gen_f = G_model(Front_FeaMap)
                gen_p = G_model(Profile_FeaMap)

                # Mapping to single unit by using Discriminator
                syn_f_gan = D_model(gen_f)
                syn_p_gan = D_model(gen_p)
                real_gan = D_model(batch_front)

                # Gradient Penalty
                gp_alpha = torch.FloatTensor(batch_front.size()[0], 1, 1, 1).to(device)
                gp_alpha.uniform_()
                interpolates = gen_p.data * gp_alpha + (1 - gp_alpha) * batch_front.data
                interpolates = interpolates.to(device).requires_grad_()  # requires_grad_() 開啟張量
                Loss, Wdis, GP = D_model.CriticWithGP_Loss(syn_f_gan, syn_p_gan, real_gan, interpolates)

                L_D = Loss
                L_D.backward()
                optimizer_D.step()
            writer.add_scalar('Discriminator/Gradient-Penalty', GP, steps)
            writer.add_scalar('Discriminator/Wasserstein-Distance', Wdis, steps)
            writer.add_scalar('Discriminator/D-LOSS', Loss, steps)
            log_learning(epoch, steps, 'D', D_lr, L_D.data, args)

            enable_gradients(G_model)
            disable_gradients(D_model)
            disable_gradients(C_model)
            for _ in range(0, args.num_critic_G):
                G_model.zero_grad()
                """Loss Functions
                    1. Pixel-Wise Loss: front-to-front reconstruct
                    2. Perceptual Loss: Feature distance on space of pretrined face model
                    3. Regulation Loss: L2 weight regulation (Aleady included in nn.Adam)
                    4. Adversarial Loss: Wasserstein Distance
                    5. Symmetric Loss: NOT APPLY
                    6. Drift Loss: NOT APPLY
                    7. Grade Penalty Loss: Grade penalty for Discriminator
                    """

                # Create Encoder Feature Map / Get the Real images' Features

                Front_Fea, Front_FeaMap = C_model(batch_front)
                Profile_Fea, Profile_FeaMap = C_model(batch_profile)

                # Synthesized image / Get the Fake images' Features
                gen_f = G_model(Front_FeaMap)
                gen_p = G_model(Profile_FeaMap)

                Front_Syn_Fea, _ = C_model(gen_f)
                Profile_Syn_Fea, _ = C_model(gen_p)

                # Mapping to single unit by using Discriminator
                syn_f_gan = D_model(gen_f)
                syn_p_gan = D_model(gen_p)

                # Frontalization Loss: L1-Norm
                L1 = G_model.L1Loss(gen_f, batch_front)  #(input, target)
                # Feature Loss: Cosine-Norm / L2-Norm
                L2 = G_model.L2Loss(Front_Syn_Fea, Front_Fea, Profile_Syn_Fea, Profile_Fea)
                # Adversarial Loss
                L_Gen = G_model.GLoss(syn_f_gan, syn_p_gan)
                # L2 Regulation Loss (L2 regularization on the parameters of the model is already included in most optimizers)

                L_G = args.lambda_l1*L1 + args.lambda_fea*L2 + args.lambda_gan*L_Gen
                L_G.backward()
                optimizer_G.step()
                writer.add_scalar('Generator/Pixel-Wise-Loss', L1, steps)
                writer.add_scalar('Generator/Perceptual-Loss', L2, steps)
                writer.add_scalar('Generator/Adversarial Loss', L_Gen, steps)
                writer.add_scalar('Generator/G-LOSS', L_Gen, steps)
                log_learning(epoch, steps, 'G', G_lr, L_G.data, args)


            if steps % 500 == 0:

                x_r = vutils.make_grid(batch_front, normalize=True, scale_each=True)
                y_r = vutils.make_grid(batch_profile, normalize=True, scale_each=True)
                x_f = vutils.make_grid(gen_f, normalize=True, scale_each=True)
                y_f = vutils.make_grid(gen_p, normalize=True, scale_each=True)
                writer.add_image('Image/Front-Real', x_r, steps)
                writer.add_image('Image/Front-Generated', x_f, steps)
                writer.add_image('Image/Profile-Real', y_r, steps)
                writer.add_image('Image/Profile-Generated', y_f, steps)

                save_path_image = os.path.join(args.snapshot_dir, 'epoch{}_FrontInput.jpg'.format(epoch))
                torchvision.utils.save_image(batch_front, save_path_image, normalize=True, scale_each=True)
                save_path_image = os.path.join(args.snapshot_dir, 'epoch{}_FrontSynthesized.jpg'.format(epoch))
                torchvision.utils.save_image(gen_f, save_path_image, normalize=True, scale_each=True)

                save_path_image = os.path.join(args.snapshot_dir, 'epoch{}_ProfileInput.jpg'.format(epoch))
                torchvision.utils.save_image(batch_profile, save_path_image, normalize=True, scale_each=True)
                save_path_image = os.path.join(args.snapshot_dir, 'epoch{}_ProfileSynthesized.jpg'.format(epoch))
                torchvision.utils.save_image(gen_p, save_path_image, normalize=True, scale_each=True)

        if epoch % args.save_freq == 0:
            if not os.path.isdir(args.snapshot_dir): os.makedirs(args.snapshot_dir)
            save_path_D = os.path.join(args.snapshot_dir, 'epoch{}_D.pt'.format(epoch))
            torch.save(D_model.state_dict(), save_path_D)
            save_path_G = os.path.join(args.snapshot_dir, 'epoch{}_G.pt'.format(epoch))
            torch.save(G_model.state_dict(), save_path_G)
            save_checkpoint({
                'epoch': epoch + 1,
                'D_model': D_model.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'G_model': G_model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
            }, save_dir=os.path.join(args.snapshot_dir, 'epoch{}'.format(epoch)))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
