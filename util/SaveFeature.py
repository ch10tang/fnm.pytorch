import os
from util.ConcatPath import ConcatPath

def SaveFeature(features, batchImgName, SaveRoot):
    for feature, ImgName in zip(features, batchImgName):
        # with will automatically close the txt file
        tmp = ImgName.split('/')
        # type = tmp[0]
        # subject = tmp[1]
        # SavePath = os.path.join(SaveRoot, type, subject)
        FeaturePath, SavePath = ConcatPath(SaveRoot, tmp, '.txt')
        if not os.path.isdir(SavePath): os.makedirs(SavePath)
        # FeaturePath = SavePath + '/' + tmp[2][:len(tmp[2])-4] + '.txt'
        with open(FeaturePath, 'w') as f:
            for fea in feature:
                text = str(fea)
                f.write("{}\n".format(text))




