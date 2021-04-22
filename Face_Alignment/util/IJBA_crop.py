import cv2
import numpy as np
import math


# Mimic LightCNN (Mouth middle eyes center = 84)
# upper_bound = new_eye_mid_y - 84 + tp_pad
# lower_bound = new_eye_mid_y + 166 + tp_pad
# left_bound = new_eye_mid_x - 125 + tp_pad
# right_bound = new_eye_mid_x + 125 + tp_pad

# Mimic WarpAffine (Mouth middle eyes center = 90)
# upper_bound = new_eye_mid_y - 112 + tp_pad
# lower_bound = new_eye_mid_y + 138 + tp_pad
# left_bound = new_eye_mid_x - 125 + tp_pad
# right_bound = new_eye_mid_x + 125 + tp_pad


def lmrk_select(lmrks):
    if len(lmrks) == 68:
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

    elif len(lmrks) == 10:
        LeftEye_x = int((lmrks[0][0] + lmrks[1][0]) / 2)
        LeftEye_y = int((lmrks[0][1] + lmrks[1][1]) / 2)
        RightEye_x = int((lmrks[2][0] + lmrks[3][0]) / 2)
        RightEye_y = int((lmrks[2][1] + lmrks[3][1]) / 2)
        EyeCenter_x = int((RightEye_x + LeftEye_x) / 2)
        EyeCenter_y = int((RightEye_y + LeftEye_y) / 2)
        LeftMouth_x = int(lmrks[5][0])
        LeftMouth_y = int(lmrks[5][1])
        RightMouth_x = int(lmrks[6][0])
        RightMouth_y = int(lmrks[6][1])
        Mouth_Center_x = int((RightMouth_x + LeftMouth_x) / 2)
        Mouth_Center_y = int((RightMouth_y + LeftMouth_y) / 2)
        Chin_x = int(lmrks[8][0])
        Chin_y = int(lmrks[8][1])

        facial5points = [[LeftEye_x, LeftEye_y], [RightEye_x, RightEye_y], [EyeCenter_x, EyeCenter_y],
                         [Mouth_Center_x, Mouth_Center_y], [Chin_x, Chin_y]]

    elif len(lmrks) == 5:

        EyeCenter_x = int(lmrks[1][0])
        EyeCenter_y = int(lmrks[0][1])
        Mouth_Center_x = int(lmrks[3][0])
        Mouth_Center_y = int(lmrks[3][1])
        Chin_x = int(lmrks[3][0])
        Chin_y = int(lmrks[3][1])

        facial5points = [[], [], [EyeCenter_x, EyeCenter_y],
                         [Mouth_Center_x, Mouth_Center_y], [Chin_x, Chin_y]]
    else:
        print('Something wrong here!')
    return facial5points


def facial_crop(img, lmrks):


    scale = 90 / (((lmrks[2][0] - lmrks[3][0]) ** 2 + (lmrks[2][1] - lmrks[3][1]) ** 2) ** 0.5)
    atan = math.atan2(lmrks[0][1] - lmrks[1][1], lmrks[0][0] - lmrks[1][0]) * 57.2957795 + 180
    center = (lmrks[2][0], lmrks[2][1])
    M = cv2.getRotationMatrix2D(center, atan, scale)
    rotated_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    upper_bound = lmrks[2][1] - 112
    lower_bound = lmrks[2][1] + 138
    left_bound = lmrks[2][0] - 125
    right_bound = lmrks[2][0] + 125

    if upper_bound < 0 or left_bound < 0 or lower_bound > rotated_face.shape[0] or right_bound > rotated_face.shape[1]:
        try:
            tmp = np.array(
                [upper_bound, rotated_face.shape[0] - lower_bound, left_bound, rotated_face.shape[1] - right_bound])
            tp_pad = abs(min(tmp[np.where(tmp < 0)]))

            rotated_face = cv2.copyMakeBorder(rotated_face, tp_pad, tp_pad, tp_pad, tp_pad, cv2.BORDER_CONSTANT,
                                              value=0)
            upper_bound = upper_bound + tp_pad
            lower_bound = lower_bound + tp_pad
            left_bound = left_bound + tp_pad
            right_bound = right_bound + tp_pad
        except:
            print('d')


    cropped_face = rotated_face[upper_bound:lower_bound, left_bound:right_bound, :]
    # cv2.imshow('d', cropped_face)
    # cv2.waitKey()

    return cropped_face


def large_pose_crop(img, lmrks):
    eye_middle_x = lmrks[2][0]
    eye_middle_y = lmrks[2][1]
    mouth_middle_y = lmrks[3][1]
    chin_y = lmrks[4][1]

    # Scale_thresh = 145 / (chin_y - eye_middle_y) # Mimic LightCNN
    Scale_thresh = 145 / (chin_y - eye_middle_y)  # Mimic LightCNN
    if Scale_thresh<0:
        return []
    resize_img = cv2.resize(img, (int(img.shape[1] * Scale_thresh), int(img.shape[0] * Scale_thresh)))
    new_eye_mid_x, new_eye_mid_y = int(eye_middle_x * Scale_thresh), int(eye_middle_y * Scale_thresh)

    upper_bound = new_eye_mid_y - 112
    lower_bound = new_eye_mid_y + 138
    left_bound = new_eye_mid_x - 125
    right_bound = new_eye_mid_x + 125

    if upper_bound < 0 or left_bound < 0 or lower_bound > resize_img.shape[0] or right_bound > resize_img.shape[1]:
        tmp = np.array([upper_bound, resize_img.shape[1] - lower_bound, left_bound, resize_img.shape[1] - right_bound])
        tp_pad = abs(min(tmp[np.where(tmp < 0)]))

        resize_img = cv2.copyMakeBorder(resize_img, tp_pad, tp_pad, tp_pad, tp_pad, cv2.BORDER_CONSTANT, value=0)
        upper_bound = new_eye_mid_y - 112 + tp_pad
        lower_bound = new_eye_mid_y + 138 + tp_pad
        left_bound = new_eye_mid_x - 125 + tp_pad
        right_bound = new_eye_mid_x + 125 + tp_pad

    cropped_face = resize_img[upper_bound:lower_bound, left_bound:right_bound, :]
    # cv2.imshow('d', cropped_face)
    # cv2.waitKey()

    return cropped_face

