import cv2
import numpy as np
from align_trans import warp_and_crop_face
import math

def facial_bound(new_eye_mid_y, new_eye_mid_x, crop_size):

    # According to
    # upper_bound = new_eye_mid_y - 50
    # lower_bound = new_eye_mid_y + 62
    # left_bound = new_eye_mid_x - 56
    # right_bound = new_eye_mid_x + 56

    upper = math.ceil(crop_size*(1/3))
    lower = crop_size - upper

    upper_bound = new_eye_mid_y - int(upper)
    lower_bound = new_eye_mid_y + int(lower)
    left_bound = new_eye_mid_x - int((crop_size/2))
    right_bound = new_eye_mid_x + int((crop_size/2))

    return upper_bound, lower_bound, left_bound, right_bound

def facial_pad(facial_boundary, resize_img):

    upper_bound = facial_boundary[0]
    lower_bound = facial_boundary[1]
    left_bound = facial_boundary[2]
    right_bound = facial_boundary[3]

    if upper_bound < 0 or left_bound < 0 or lower_bound > resize_img.shape[0] or right_bound > resize_img.shape[1]:
        tmp = np.array([upper_bound, resize_img.shape[1] - lower_bound, left_bound, resize_img.shape[1] - right_bound])
        tp_pad = abs(min(tmp[np.where(tmp < 0)]))

        resize_img = cv2.copyMakeBorder(resize_img, tp_pad, tp_pad, tp_pad, tp_pad, cv2.BORDER_CONSTANT, value=0)
        upper_bound = upper_bound + tp_pad
        lower_bound = lower_bound + tp_pad
        left_bound = left_bound + tp_pad
        right_bound = right_bound + tp_pad

    return [upper_bound, lower_bound, left_bound, right_bound], resize_img

def large_pose_crop(img, lmrks, pose, reference, args):

    scale_size = math.ceil(args.crop_size / 144 * 48)  # According to LightCNN cropping strategy

    if args.WarpAffine:
        warped_face = warp_and_crop_face(np.array(img), lmrks, reference, crop_size=(args.crop_size, args.crop_size))

    elif args.Align_Eyes_Crop:
        if pose == '110' or pose == '120' or pose == '010' or pose == '240':
            eye_middle_x = (lmrks[0][0] + lmrks[1][0]) / 2
            eye_middle_y = (lmrks[0][1] + lmrks[1][1]) / 2
            mouth_middle_y = (lmrks[3][1] + lmrks[4][1]) / 2

            Scale_thresh = scale_size / (mouth_middle_y - eye_middle_y)
            resize_img = cv2.resize(img, (int(img.shape[1] * Scale_thresh), int(img.shape[0] * Scale_thresh)))
            new_eye_mid_x, new_eye_mid_y = int(eye_middle_x * Scale_thresh), int(eye_middle_y * Scale_thresh)

            facial_boundary = facial_bound(new_eye_mid_y, new_eye_mid_x, args.crop_size)
            [upper_bound, lower_bound, left_bound, right_bound], img = facial_pad(facial_boundary, resize_img)

        else:
            scale = scale_size / (((lmrks[2][0] - lmrks[3][0]) ** 2 + (lmrks[2][1] - lmrks[3][1]) ** 2) ** 0.5)
            atan = math.atan2(lmrks[0][1] - lmrks[1][1],lmrks[0][0] - lmrks[1][0]) * 57.2957795 + 180
            center = (lmrks[2][0], lmrks[2][1])
            M = cv2.getRotationMatrix2D(center, atan, scale)
            rotated_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            facial_boundary = facial_bound(lmrks[2][1], lmrks[2][0], args.crop_size)
            [upper_bound, lower_bound, left_bound, right_bound], img = facial_pad(facial_boundary, rotated_face)

        warped_face = img[upper_bound:lower_bound, left_bound:right_bound, :]

    return warped_face

