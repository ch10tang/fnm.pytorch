import numpy as np
import cv2
import os
import winsound

def Draw_GT_Line(landmarks, args):

    for idx in range(0, len(landmarks)):

        img = np.zeros((args.crop_size, args.crop_size, 1))
        line_color = (255, 255, 255)
        line_width = 2
        for n in range(0, 16):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n+1][0])), int(float(landmarks[n+1][1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(31, 35):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))), (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))), (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)
        for n in range(48, 59):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[48][0])), int(float(landmarks[48][1]))), (int(float(landmarks[59][0])), int(float(landmarks[59][1]))), line_color, line_width)
        for n in range(60, 67):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))), (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[60][0])), int(float(landmarks[60][1]))), (int(float(landmarks[67][0])), int(float(landmarks[67][1]))), line_color, line_width)

    img = cv2.merge([img, img, img])

    return img

