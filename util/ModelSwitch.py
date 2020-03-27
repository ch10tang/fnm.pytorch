import torch

def ModelSwith(D_model, G_model):
    for s1, s2 in zip(D_model.convLayers.parameters(), G_model.G_enc_convLayers.parameters()):
        s1 = s2
