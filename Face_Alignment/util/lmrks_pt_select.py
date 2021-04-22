


def lmrks_pt_select(lmrks, args, Specific=None):

    if args.WarpAffine:
        LeftEye_x = int((lmrks[36][0] + lmrks[39][0]) / 2)
        LeftEye_y = int((lmrks[36][1] + lmrks[39][1]) / 2)
        RightEye_x = int((lmrks[42][0] + lmrks[45][0]) / 2)
        RightEye_y = int((lmrks[42][1] + lmrks[45][1]) / 2)
        Noise_x = int(lmrks[30][0])
        Noise_y = int(lmrks[30][1])
        LeftMouth_x = int(lmrks[48][0])
        LeftMouth_y = int(lmrks[48][1])
        RightMouth_x = int(lmrks[54][0])
        RightMouth_y = int(lmrks[54][1])
        facial5points = [[LeftEye_x, LeftEye_y], [RightEye_x, RightEye_y], [Noise_x, Noise_y],
                         [LeftMouth_x, LeftMouth_y], [RightMouth_x, RightMouth_y]]

    elif args.Align_Eyes_Crop:
        LeftEye_x = int((lmrks[36][0] + lmrks[39][0]) / 2)
        LeftEye_y = int((lmrks[36][1] + lmrks[39][1]) / 2)
        RightEye_x = int((lmrks[42][0] + lmrks[45][0]) / 2)
        RightEye_y = int((lmrks[42][1] + lmrks[45][1]) / 2)
        EyeCenter_x = int((RightEye_x + LeftEye_x) / 2)
        EyeCenter_y = int((RightEye_y + LeftEye_y) / 2)
        LeftMouth_x = int(lmrks[48][0])
        LeftMouth_y = int(lmrks[48][1])
        RightMouth_x = int(lmrks[54][0])
        RightMouth_y = int(lmrks[54][1])
        Mouth_Center_x = int((RightMouth_x + LeftMouth_x) / 2)
        Mouth_Center_y = int((RightMouth_y + LeftMouth_y) / 2)
        Chin_x = int(lmrks[8][0])
        Chin_y = int(lmrks[8][1])


        facial5points = [[LeftEye_x, LeftEye_y], [RightEye_x, RightEye_y], [EyeCenter_x, EyeCenter_y],
                        [Mouth_Center_x, Mouth_Center_y], [Chin_x, Chin_y]]
    elif Specific=='IJB-A':
        print('hjhle')


    return facial5points