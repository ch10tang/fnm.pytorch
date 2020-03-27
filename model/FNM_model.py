import torch
from torch import nn
from torch.autograd import grad


loss_criterion = nn.CrossEntropyLoss().cuda()
loss_criterion_L1 = nn.L1Loss(reduction='mean').cuda()
loss_criterion_L2 = nn.MSELoss(reduction='mean').cuda()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # torch.nn.Conv2d(in, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        if args.ArcFace:
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
        elif args.VGGFace2:
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
        self.res1_1 = self.res_block(512)
        self.res1_2 = self.res_block(512)
        self.res1_3 = self.res_block(512)
        self.res1_4 = self.res_block(512)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.res2 = self.res_block(256)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.res3 = self.res_block(128)
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.res4 = self.res_block(64)
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.res5 = self.res_block(32)
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.res6 = self.res_block(32)
        self.deconv7 = nn.Sequential(nn.ConvTranspose2d(32, 3, 3, 1, 1), nn.Tanh())
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    # def res_block(self, output_channles, kernels=3, stride=1, padding=1):
    def res_block(self, output_channles, kernels=3, stride=1, padding=1):

        convLayers = [
            nn.Conv2d(output_channles, output_channles, kernels, stride, padding),
            nn.BatchNorm2d(output_channles),
            nn.ReLU(),
            nn.Conv2d(output_channles, output_channles, 3, 1, 1),
            nn.BatchNorm2d(output_channles),
            nn.ReLU(),
        ]
        return nn.Sequential(*convLayers)

    def forward(self, input):

        x = self.deconv1(input)
        x = self.relu(self.res1_1(x) + x)
        x = self.relu(self.res1_2(x) + x)
        x = self.relu(self.res1_3(x) + x)
        x = self.relu(self.res1_4(x) + x)
        x = self.deconv2(x)
        x = self.relu(self.res2(x) + x)
        x = self.deconv3(x)
        x = self.relu(self.res3(x) + x)
        x = self.deconv4(x)
        x = self.relu(self.res4(x) + x)
        x = self.deconv5(x)
        x = self.relu(self.res5(x) + x)
        x = self.deconv6(x)
        x = self.relu(self.res6(x) + x)
        output = self.deconv7(x)

        return (output + 1) * 127.5

    def L1Loss(self, input, target):

        Loss = loss_criterion_L1(input, target)  # L1Loss(input, target)

        return Loss

    def L2Loss(self, f_in, f_tgt, p_in, p_tgt):

        epsilon = 1e-9
        #
        f_in = torch.div(f_in, (f_in.norm(2, dim=1, keepdim=True) + epsilon))
        f_tgt = torch.div(f_tgt, (f_tgt.norm(2, dim=1, keepdim=True) + epsilon))
        p_in = torch.div(p_in, (p_in.norm(2, dim=1, keepdim=True) + epsilon))
        p_tgt = torch.div(p_tgt, (p_tgt.norm(2, dim=1, keepdim=True) + epsilon))
        #
        loss = ((1 - torch.mul(f_in, f_tgt).sum(1))*0.5 + (1 - torch.mul(p_in, p_tgt).sum(1))*0.5).sum()
        #
        return loss

        # f_loss = loss_criterion_L2(f_in, f_tgt)
        # p_loss = loss_criterion_L2(p_in, p_tgt)

        # return (f_loss + p_loss)/2

    def GLoss(self, Syn_F_GAN, Syn_P_GAN):

        Syn_F = Syn_F_GAN[0] + Syn_F_GAN[1] + Syn_F_GAN[2] + Syn_F_GAN[3] + Syn_F_GAN[4]
        Syn_P = Syn_P_GAN[0] + Syn_P_GAN[1] + Syn_P_GAN[2] + Syn_P_GAN[3] + Syn_P_GAN[4]

        loss = -(Syn_F*0.5 + Syn_P*0.5).mean() / 5

        return loss

    def RegLoss(self):

        reg_loss = 0
        for param in self.modules():
            reg_loss += l1_crit(param)

        factor = 0.0005
        loss = factor * reg_loss

        return loss

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.features = []
        self.Channel = args.Channel
        self.lmbda_gp = args.lmbda_gp
        # torch.nn.Conv2d(in, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        image_connLayers = [
            nn.Conv2d(self.Channel, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([56, 56]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([28, 28]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([14, 14]),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([7, 7]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(12544, 1),
        ]
        eyes_convLayers = [
            nn.Conv2d(self.Channel, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([12, 34]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([6, 17]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([3, 9]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(6912, 1),
        ]
        nose_convLayers = [
            nn.Conv2d(self.Channel, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([19, 12]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([10, 6]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([5, 3]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(3840, 1),
        ]
        mouth_convLayers = [
            nn.Conv2d(self.Channel, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([9, 16]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([5, 8]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([3, 4]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(3072, 1),
        ]
        face_convLayers = [
            nn.Conv2d(self.Channel, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([38, 39]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([19, 20]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([10, 10]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(25600, 1),
        ]

        self.eyes_convLayers = nn.Sequential(*eyes_convLayers)
        self.nose_convLayers = nn.Sequential(*nose_convLayers)
        self.mouth_convLayers = nn.Sequential(*mouth_convLayers)
        self.face_convLayers = nn.Sequential(*face_convLayers)
        self.image_connLayers = nn.Sequential(*image_connLayers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            if isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)


    def slice(self, input):

        eyes = input[0:len(input), :, 56:56 + 46, 44:44 + 136]
        nose = input[0:len(input), :, 70:70 + 74, 88:88 + 48]
        mouth = input[0:len(input), :, 144:144 + 36, 80:80 + 64]
        face = input[0:len(input), :, 40:40 + 150, 34:34 + 156]

        return [eyes, nose, mouth, face]

    def Disc(self, input):

        input = (input / 127.5) - 1
        eyes_ROI, nose_ROI, mouth_ROI, face_ROI = self.slice(input)

        input = self.image_connLayers(input)
        eyes = self.eyes_convLayers(eyes_ROI)
        nose = self.nose_convLayers(nose_ROI)
        mouth = self.mouth_convLayers(mouth_ROI)
        face = self.face_convLayers(face_ROI)

        return input, eyes, nose, mouth, face

    def forward(self, input):

        output = self.Disc(input)

        return output

    def CriticWithGP_Loss(self, Syn_F_Gan, Syn_P_Gan, Real_Gan, Interpolates):


        Syn_F = Syn_F_Gan[0] + Syn_F_Gan[1] + Syn_F_Gan[2] + Syn_F_Gan[3] + Syn_F_Gan[4]
        Syn_P = Syn_P_Gan[0] + Syn_P_Gan[1] + Syn_P_Gan[2] + Syn_P_Gan[3] + Syn_P_Gan[4]
        Real = Real_Gan[0] + Real_Gan[1] + Real_Gan[2] + Real_Gan[3] + Real_Gan[4]
        Wasserstein_Dis = (Syn_F*0.5 + Syn_P*0.5 - Real).mean() / 5

        inter = self.Disc(Interpolates)
        gradinput = grad(outputs=inter[0].sum(), inputs=Interpolates, create_graph=True)[0]
        gradeyes = grad(outputs=inter[1].sum(), inputs=Interpolates, create_graph=True)[0]
        gradnose = grad(outputs=inter[2].sum(), inputs=Interpolates, create_graph=True)[0]
        gradmouth = grad(outputs=inter[3].sum(), inputs=Interpolates, create_graph=True)[0]
        gradface = grad(outputs=inter[4].sum(), inputs=Interpolates, create_graph=True)[0]
        gradients = gradinput + gradeyes + gradnose + gradmouth + gradface

        # calculate gradient penalty
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

        loss = Wasserstein_Dis + self.lmbda_gp * gradient_penalty

        return loss, Wasserstein_Dis, gradient_penalty

