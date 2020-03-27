from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from numpy.linalg import norm
import torch.nn.functional as F
from torch.autograd import Variable
import torch


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
        ID = self.imgFrame.ix[idx, 1]
        image = self.transform(image)

        return [image, imgName, ID]

def PoseSelect(PoseName):

    if PoseName == 'frontal':
        pose = 0
    elif PoseName == 'profile':
        pose = 1
    else:
        print('Something wrong!')
        exit()

    return pose

def ConcatPath(SaveRoot, ImgName, str):

    for ii in range(len(ImgName)-1):
        SaveRoot = os.path.join(SaveRoot, ImgName[ii])

    name = ImgName[len(ImgName)-1]
    SavePath = os.path.join(SaveRoot, name[:len(name)-4] + str)

    return SavePath, SaveRoot


def Validation_CFP(G_model, OTS_C_Model, args):

    G_model.eval()
    OTS_C_Model.eval()


    # Load augmented data
    transformed_dataset = FaceIdPoseDataset(args.val_csv_file, '{}/CFP/CFP_FOCropImage_DIM112x112_NameReady_v1'.format(args.data_place),
                                            transform=transforms.Compose([torchvision.transforms.Resize(256),
                                                                          torchvision.transforms.CenterCrop(224),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                                                               (0.5, 0.5, 0.5))]))
    dataloader = DataLoader(transformed_dataset, batch_size=10, shuffle=False)  # , num_workers=6)

    Features_List = {'Subject':[], 'Pose':[], 'ImgNum':[], 'Features':[]}
    count = 0

    for i, batch_data in enumerate(dataloader):

        minibatch_size = len(batch_data[0])
        yaw_code_label_frontal = torch.LongTensor(np.ones(minibatch_size) * 6)
        yaw_code_frontal = torch.FloatTensor(one_hot(yaw_code_label_frontal, Ny))
        fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))

        if args.cuda:
            batch_image, yaw_code_frontal, fixed_noise = batch_data[0].cuda(), yaw_code_frontal.cuda(), fixed_noise.cuda()
        batch_image, yaw_code_frontal, fixed_noise = Variable(batch_image), Variable(yaw_code_frontal), Variable(fixed_noise)



        generated = G_model(batch_image, yaw_code_frontal, fixed_noise)
        if OTS:
            generated = F.interpolate(generated, 112, mode='bilinear',align_corners=False)
        _ = OTS_C_Model(generated)

        features = (OTS_C_Model.features.data).cpu().numpy()
        batchImageName = batch_data[1]


        for ImgName, feas in zip(batchImageName, features):

            tmp = ImgName.split('/')
            Pose = PoseSelect(tmp[1])

            Features_List['Subject'].append(int(tmp[0]))
            Features_List['Pose'].append(Pose)
            Features_List['ImgNum'].append(int(tmp[2].split('.')[0]))
            Features_List['Features'].append(feas)

        count += 10
        print("Finish Processing {} images...".format(count))


    print('Loading the CFP protocol ...')

    PairRoot = './Protocol/Split2Mat'
    Type = ['FF', 'FP']
    TotalMean = []
    TotalStd = []
    ACCURACY = []
    ACCURACY = np.array(ACCURACY)

    Fea = pd.DataFrame.from_dict(Features_List)
    for tp in range(len(Type)):
        SplitList = os.listdir(os.path.join(PairRoot, Type[tp]))
        for s1 in range(0, len(SplitList)):
            Name = ['same.mat', 'diff.mat']
            DISTANCE = []
            LABEL = []
            for nn in range(2):
                DataName = '{}/{}/{}/{}'.format(PairRoot, Type[tp], SplitList[s1], Name[nn])
                data = sio.loadmat(DataName)#,  struct_as_record=False)

                Path = data['Pair'][0, 0]['Path']


                for pp in Path:

                    tmp1 = pp[0][0].split('/')
                    tmp2 = pp[1][0].split('/')

                    Pose = PoseSelect(tmp1[4])
                    Fea1 = Fea.Features[Fea[(Fea.Subject == int(tmp1[3]))&(Fea.Pose == Pose)&
                                            (Fea.ImgNum == int(tmp1[5].split('.')[0]))].index.tolist()[0]]

                    Pose = PoseSelect(tmp2[4])
                    Fea2 = Fea.Features[Fea[(Fea.Subject == int(tmp2[3])) & (Fea.Pose == Pose) &
                                            (Fea.ImgNum == int(tmp2[5].split('.')[0]))].index.tolist()[0]]

                    if len(Fea[(Fea.Subject == int(tmp1[3]))&(Fea.Pose == Pose)&(Fea.ImgNum == int(tmp1[5].split('.')[0]))].index.tolist())>1 or len(Fea[(Fea.Subject == int(tmp2[3]))&(Fea.Pose == Pose)&(Fea.ImgNum == int(tmp2[5].split('.')[0]))].index.tolist())>1:
                        exit()


                    distance = cdist(Fea1.reshape(1, -1) / norm(Fea1.reshape(1, -1)),
                                     Fea2.reshape(1, -1) / norm(Fea2.reshape(1, -1)), 'cosine')
                    DISTANCE.append(distance)
                LABEL.append(data['Pair'][0, 0]['Label'])

            DISTANCE_array = np.array(DISTANCE).reshape(-1, 1)
            LABEL_array = np.array(LABEL).reshape(-1, 1)
            Result_TAR = []
            Result_FAR = []
            Result_TAR = np.array(Result_TAR)
            Result_FAR = np.array(Result_FAR)
            Result_BestAcc = 0
            for thresh in range(0, 151):
                thresh = thresh / 100
                THRESH = np.ones([len(LABEL_array), 1]) * thresh
                Intra_predict = [pre_indx for (pre_indx, val) in enumerate(THRESH[0:350] - DISTANCE_array[0:350]) if
                                 val > 0]
                extra_predict = [pre_exdx for (pre_exdx, val2) in enumerate(THRESH[350:700] - DISTANCE_array[350:700])
                                 if val2 < 0]
                Result_TAR = np.append(Result_TAR, len(Intra_predict) / 350)
                Result_FAR = np.append(Result_FAR, (350 - len(extra_predict)) / 350)
                ACC = (len(Intra_predict) + len(extra_predict)) / 700
                ACC = float('%.4f' % ACC)
                if ACC > Result_BestAcc:
                    Result_BestAcc = ACC

            ACCURACY = np.append(ACCURACY, Result_BestAcc)
            print('>>> Split {} \tBest Accuracy : {} {} \n'.format(s1, Result_BestAcc * 100, '%'))

        STD = np.std(ACCURACY, ddof=1)
        STD = round(STD, 4)
        MEAN = np.mean(ACCURACY)
        MEAN = float('%.4f' % MEAN)
        print('>>> CFP {} Mean Accuracy : {}{}, Std={}\n'.format(Type[tp], MEAN * 100, '%', STD * 100))
        ACCURACY = np.array([])

        TotalMean.append(MEAN)
        TotalStd.append(STD)

    return TotalMean, TotalStd





