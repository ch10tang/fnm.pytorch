from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
import argparse
import cv2
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument("-source_root", help="specify your source dir", default="../Eval/Src_Img", type = str)
    parser.add_argument("-dest_root", help="specify your destination dir", default="../Eval/Face_Cropped", type = str)
    parser.add_argument("-crop_size", help="specify size of aligned faces, align and crop with padding", default=250, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.makedirs(dest_root)

    count = 0
    for idx, (roots, dirs, files) in enumerate(os.walk(source_root)):

        for file in files:
            if file=='*.DS_Store':
                continue
        
            img = Image.open(os.path.join(roots, file))
            try: # Handle exception
                _, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(roots, file)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                print("{} is discarded due to non-detected landmarks!".format(os.path.join(roots, file)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            img_warped.save(os.path.join(dest_root, file))

    print('Done!')