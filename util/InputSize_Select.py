import torch.nn.functional as F

def InputSize_Select(input, args):

    if args.model_select =='VGGFace2':
        input = input

    elif args.model_select == 'Light_CNN_9' or args.model_select == 'Light_CNN_29' or args.model_select == 'Light_CNN_29_v2':
        input = F.interpolate(input, 128, mode='bilinear', align_corners=False)
        input = (input / 127.5) - 1
        input = input[:, 0, :, :].unsqueeze(1)

    elif args.model_select == 'IR-50':
        input = F.interpolate(input, 112, mode='bilinear', align_corners=False)
        input = (input / 127.5) -1

    else:
        print('Please select valid pretrained model !')
        exit()

    return input