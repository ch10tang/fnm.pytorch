from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from util.GenerateCodeLabel import GenerateCodeLabel
from util.one_hot import one_hot
import torchvision.utils as vutils


def Validation_Process(C_model, D_model, G_model, epoch, writer, args):


    C_model.eval()
    G_model.eval()

    Nd = args.Nd
    Ny = args.Ny
    Nz = args.Nz
    AngleLoss = args.Angle_Loss

    Flag=True
    print("Start Validating...")

    validation_dataset = FaceIdPoseDataset(args.val_csv_file, args.data_place, Flag=args.FlagEnable,
                                            transform=transforms.Compose([torchvision.transforms.Resize(110),
                                                                          transforms.CenterCrop(96),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    if not(args.multi_DRGAN):

        ID_Real_Precision = []
        ID_Fake_Precision = []
        Frontal_ID_Precision = []
        Yaw_Real_Precision = []
        Yaw_Fake_Precision = []


        for i, batch_data in enumerate(dataloader):
            batch_image = batch_data[0][:, [2, 1, 0], :, :]
            batch_id_label = batch_data[2]
            batch_yaw_label = batch_data[3]
            minibatch_size = len(batch_image)

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))
            yaw_code, yaw_code_label = GenerateCodeLabel(Ny, minibatch_size=minibatch_size)
            yaw_code_label_frontal = torch.LongTensor(np.ones(minibatch_size) * 6)
            yaw_code_frontal = torch.FloatTensor(one_hot(yaw_code_label_frontal, Ny))

            if args.cuda:
                batch_image, batch_id_label, batch_yaw_label, yaw_code_frontal, yaw_code_label_frontal = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_yaw_label.cuda(),  yaw_code_frontal.cuda(), yaw_code_label_frontal.cuda()

                fixed_noise, yaw_code, yaw_code_label = \
                    fixed_noise.cuda(), yaw_code.cuda(), yaw_code_label.cuda()

            with torch.no_grad():
                batch_image, batch_id_label, batch_yaw_label, yaw_code_frontal, yaw_code_label_frontal = \
                    Variable(batch_image), Variable(batch_id_label), Variable(batch_yaw_label), \
                    Variable(yaw_code_frontal), Variable(yaw_code_label_frontal)

                fixed_noise, yaw_code, yaw_code_label = \
                    Variable(fixed_noise), Variable(yaw_code), Variable(yaw_code_label)

            generated = G_model(batch_image, yaw_code, fixed_noise)
            generated_frontal = G_model(batch_image, yaw_code_frontal, fixed_noise)

            if Flag==True:
                x = vutils.make_grid(generated.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                y = vutils.make_grid(batch_image.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                z = vutils.make_grid(generated_frontal.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                writer.add_image('Image/Validation-Real', y, epoch)
                writer.add_image('Image/Validation-Generated', x, epoch)
                writer.add_image('Image/Validation-Generated-frontal', z, epoch)
                Flag = False

            real_output = C_model(batch_image)
            syn_output = C_model(generated)
            syn_output_frontal = C_model(generated_frontal)

            if AngleLoss:
                _, id_real_ans = torch.max(real_output[0][:, :Nd], 1)
                _, yaw_real_ans = torch.max(real_output[0][:, Nd:Nd + Ny], 1)
                _, id_syn_ans = torch.max(syn_output[0][:, :Nd], 1)
                _, yaw_syn_ans = torch.max(syn_output[0][:, Nd:Nd + Ny], 1)
                _, id_syn_frontal_ans = torch.max(syn_output_frontal[0][:, :Nd], 1)

                id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / \
                                    real_output[0].size()[0]
                id_syn_precision = (id_syn_ans == batch_id_label).type(torch.FloatTensor).sum() / syn_output[0].size()[
                    0]
                yaw_real_precision = (yaw_real_ans == batch_yaw_label).type(torch.FloatTensor).sum() / \
                                     real_output[0].size()[0]
                yaw_fake_precision = (yaw_syn_ans == yaw_code_label).type(torch.FloatTensor).sum() / \
                                     syn_output[0].size()[0]
                id_syn_frontal_precision = (id_syn_frontal_ans == batch_id_label).type(torch.FloatTensor).sum() / \
                                           syn_output_frontal[0].size()[0]
            else:
                _, id_real_ans = torch.max(real_output[:, :Nd], 1)
                _, yaw_real_ans = torch.max(real_output[:, Nd:Nd + Ny], 1)
                _, id_syn_ans = torch.max(syn_output[:, :Nd], 1)
                _, yaw_syn_ans = torch.max(syn_output[:, Nd:Nd + Ny], 1)
                _, id_syn_frontal_ans = torch.max(syn_output_frontal[:, :Nd], 1)

                id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / real_output.size()[
                    0]
                id_syn_precision = (id_syn_ans == batch_id_label).type(torch.FloatTensor).sum() / syn_output.size()[0]
                yaw_real_precision = (yaw_real_ans == batch_yaw_label).type(torch.FloatTensor).sum() / \
                                     real_output.size()[0]
                yaw_fake_precision = (yaw_syn_ans == yaw_code_label).type(torch.FloatTensor).sum() / syn_output.size()[
                    0]
                id_syn_frontal_precision = (id_syn_frontal_ans == batch_id_label).type(torch.FloatTensor).sum() / \
                                           syn_output_frontal.size()[0]












            ID_Real_Precision.append(id_real_precision.data.cpu().numpy())
            ID_Fake_Precision.append(id_syn_precision.data.cpu().numpy())
            Yaw_Real_Precision.append(yaw_real_precision.data.cpu().numpy())
            Yaw_Fake_Precision.append(yaw_fake_precision.data.cpu().numpy())
            Frontal_ID_Precision.append(id_syn_frontal_precision.data.cpu().numpy())


        ID_Real_Precisions = sum(ID_Real_Precision)/len(ID_Real_Precision)
        ID_Fake_Precisions= sum(ID_Fake_Precision) / len(ID_Fake_Precision)
        Yaw_Real_Precisions = sum(Yaw_Real_Precision) / len(Yaw_Real_Precision)
        Yaw_Fake_Precisions = sum(Yaw_Fake_Precision) / len(Yaw_Fake_Precision)
        Frontal_ID_Precisions = sum(Frontal_ID_Precision) / len(Frontal_ID_Precision)

        writer.add_scalar('Accuracy/ID_Real_Precisions', ID_Real_Precisions, epoch)
        writer.add_scalar('Accuracy/Frontal_ID_Fake_Precisions', Frontal_ID_Precisions, epoch)
        writer.add_scalar('Accuracy/Yaw_Real_Precision', Yaw_Real_Precisions, epoch)
        writer.add_scalar('Accuracy/Yaw_Fake_Precision', Yaw_Fake_Precisions, epoch)
        writer.add_scalar('Accuracy/ID_Fake_Precisions', ID_Fake_Precisions, epoch)

        del batch_image, batch_id_label, batch_yaw_label, yaw_code_frontal, yaw_code_label_frontal, fixed_noise, yaw_code, yaw_code_label

    else:

        ID_Real_Precision = []
        ID_Fake_Precision = []
        Yaw_Real_Precision = []
        Yaw_Fake_Precision = []
        ID_Unique_Fake_Precision = []
        Yaw_Unique_Fake_Precision = []

        for i, batch_data in enumerate(dataloader):

            batch_image = torch.FloatTensor(batch_data[0].float())
            batch_id_label = batch_data[2]
            batch_id_label_unique = torch.LongTensor(batch_id_label[::args.Image_IDNum])
            batch_yaw_label = batch_data[3]
            minibatch_size = len(batch_image)
            minibatch_size_unique = len(batch_image) // args.Image_IDNum

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))
            yaw_code, yaw_code_label = GenerateCodeLabel(Ny, minibatch_size=minibatch_size)

            fixed_noise_unique = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size_unique, Nz)))
            yaw_code_unique, yaw_code_label_unique = GenerateCodeLabel(Ny, minibatch_size=minibatch_size_unique)

            if args.cuda:
                batch_image, batch_id_label, batch_yaw_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_yaw_label.cuda()

                fixed_noise, yaw_code, yaw_code_label = \
                    fixed_noise.cuda(), yaw_code.cuda(), yaw_code_label.cuda()

                batch_id_label_unique, fixed_noise_unique, yaw_code_unique, yaw_code_label_unique = \
                    batch_id_label_unique.cuda(), fixed_noise_unique.cuda(), yaw_code_unique.cuda(), yaw_code_label_unique.cuda()

            with torch.no_grad():
                batch_image, batch_id_label, batch_yaw_label = \
                    Variable(batch_image), Variable(batch_id_label), Variable(batch_yaw_label)

                fixed_noise,  yaw_code, yaw_code_label = \
                    Variable(fixed_noise), Variable(yaw_code), Variable(yaw_code_label)

                batch_id_label_unique, fixed_noise_unique, yaw_code_unique, yaw_code_label_unique = \
                    Variable(batch_id_label_unique), Variable(fixed_noise_unique), Variable(yaw_code_unique), Variable(yaw_code_label_unique)

                generated_unique = G_model(batch_image, yaw_code_unique, fixed_noise_unique)
                generated = G_model(batch_image, yaw_code, fixed_noise, single=True)


            if Flag==True:
                x = vutils.make_grid(generated.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                x_unique = vutils.make_grid(generated_unique.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                y = vutils.make_grid(batch_image.data[:, [2, 1, 0], :, :], normalize=True, scale_each=True)
                writer.add_image('Image/Validation-Real', y, epoch)
                writer.add_image('Image/Validation-Generated-Single', x, epoch)
                writer.add_image('Image/Validation-Generated-Multi', x_unique, epoch)
                Flag = False

            real_output = C_model(batch_image)
            syn_output = C_model(generated)
            syn_output_unique = C_model(generated_unique)

            _, id_real_ans = torch.max(real_output[0][:, :Nd], 1)
            _, yaw_real_ans = torch.max(real_output[0][:, Nd:Nd + Ny], 1)
            _, id_syn_ans = torch.max(syn_output[0][:, :Nd], 1)
            _, yaw_syn_ans = torch.max(syn_output[0][:, Nd:Nd + Ny], 1)
            _, id_syn_ans_unique = torch.max(syn_output_unique[0][:, :Nd], 1)
            _, yaw_syn_ans_unique = torch.max(syn_output_unique[0][:, Nd:Nd + Ny], 1)

            id_real_precision = (id_real_ans == batch_id_label).type(torch.FloatTensor).sum() / real_output.size()[0]
            id_syn_precision = (id_syn_ans == batch_id_label).type(torch.FloatTensor).sum() / syn_output.size()[0]
            yaw_real_precision = (yaw_real_ans == batch_yaw_label).type(torch.FloatTensor).sum() / real_output.size()[0]
            yaw_fake_precision = (yaw_syn_ans == yaw_code_label).type(torch.FloatTensor).sum() / syn_output.size()[0]
            id_unique_syn_precision = (id_syn_ans_unique == batch_id_label_unique).type(torch.FloatTensor).sum() / syn_output_unique.size()[0]
            yaw_unique_fake_precision = (yaw_syn_ans_unique == yaw_code_label_unique).type(torch.FloatTensor).sum() / syn_output_unique.size()[0]

            ID_Real_Precision.append(id_real_precision.data.cpu().numpy())
            ID_Fake_Precision.append(id_syn_precision.data.cpu().numpy())
            Yaw_Real_Precision.append(yaw_real_precision.data.cpu().numpy())
            Yaw_Fake_Precision.append(yaw_fake_precision.data.cpu().numpy())
            ID_Unique_Fake_Precision.append(id_unique_syn_precision.data.cpu().numpy())
            Yaw_Unique_Fake_Precision.append(yaw_unique_fake_precision.data.cpu().numpy())

        ID_Real_Precisions = sum(ID_Real_Precision) / len(ID_Real_Precision)
        ID_Fake_Precisions = sum(ID_Fake_Precision) / len(ID_Fake_Precision)
        Yaw_Real_Precisions = sum(Yaw_Real_Precision) / len(Yaw_Real_Precision)
        Yaw_Fake_Precisions = sum(Yaw_Fake_Precision) / len(Yaw_Fake_Precision)
        ID_Unique_Fake_Precisions = sum(ID_Unique_Fake_Precision) / len(ID_Unique_Fake_Precision)
        Yaw_Unique_Fake_Precisions = sum(Yaw_Unique_Fake_Precision) / len(Yaw_Unique_Fake_Precision)

        writer.add_scalar('Accuracy/ID_Real_Precisions', ID_Real_Precisions, epoch)
        writer.add_scalar('Accuracy/ID_Fake_Precisions', ID_Fake_Precisions, epoch)
        writer.add_scalar('Accuracy/Yaw_Real_Precision', Yaw_Real_Precisions, epoch)
        writer.add_scalar('Accuracy/Yaw_Fake_Precision', Yaw_Fake_Precisions, epoch)
        writer.add_scalar('Accuracy/ID_Unique_Fake_Precisions', ID_Unique_Fake_Precisions, epoch)
        writer.add_scalar('Accuracy/Yaw_Unique_Fake_Precisions', Yaw_Unique_Fake_Precisions, epoch)
