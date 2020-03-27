import os

def ConcatPath(SaveRoot, ImgName, str):

    for ii in range(len(ImgName)-1):
        SaveRoot = os.path.join(SaveRoot, ImgName[ii])

    name = ImgName[len(ImgName)-1]
    SavePath = os.path.join(SaveRoot, name[:len(name)-4] + str)

    return SavePath, SaveRoot