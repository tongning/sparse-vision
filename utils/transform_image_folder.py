import argparse
import glob, os
import numpy as np
from PIL import Image
import cv2
import sys
from scipy import ndimage
from tqdm import tqdm
from random import randint
from skimage.feature import hog
from skimage import data, exposure
np.set_printoptions(threshold=sys.maxsize)
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--folderpath', action='store', type=str, default="../../tiny-imagenet-200-sift")
arg_parser.add_argument('--transform', action='store', type=str, required=True)
args = arg_parser.parse_args()

root_folder = args.folderpath

sift = cv2.xfeatures2d.SIFT_create()
num_skipped = 0
for filename in tqdm(glob.iglob(os.path.join(root_folder, '**'), recursive=True)):
    if os.path.isfile(filename): # filter dirs
        if '.JPEG' in filename:
            try:
                if args.transform == 'sift':
                    id = randint(0,1000)
                    cv_img = cv2.imread(filename)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = sift.detectAndCompute(cv_img, None)
                    keypoints_list = cv2.KeyPoint_convert(keypoints).astype(int)
                    keypoints_list[:, 0], keypoints_list[:, 1] = keypoints_list[:, 1], keypoints_list[:, 0].copy()

                    keypoints_mask = np.zeros(gray.shape)
                    keypoints_mask[[*keypoints_list.T]] = 1
                    keypoints_mask = keypoints_mask.astype(bool)
                    struct = ndimage.generate_binary_structure(2, 2)
                    dilated_mask = ndimage.binary_dilation(keypoints_mask, structure=struct, iterations=3).astype(keypoints_mask.dtype)
                    mask_3d = np.stack((dilated_mask, dilated_mask, dilated_mask), axis=-1)
                    masked_image = np.multiply(cv_img, mask_3d)
                    out_img = Image.fromarray(masked_image)
                    os.remove(filename)
                    out_img.save(filename)
                elif args.transform == 'hog':
                    id = randint(0,1000)
                    cv_img = cv2.imread(filename)

                    fd, hog_image = hog(cv_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
                    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                    out_img = np.stack(   (hog_image_rescaled, hog_image_rescaled, hog_image_rescaled), axis=-1)
                    out_img *= 255
                    out_img = out_img.astype(np.uint8)
                    out_img = Image.fromarray(    out_img    )
                    os.remove(filename)
                    out_img.save(filename)
                else:
                    print("Unrecognized transform")
                    exit(0)
            except Exception as e:
                num_skipped += 1
                print(f'Skipping {filename} because of error {e}')
                print(f'Total skipped: {num_skipped}')
                os.remove(filename)
            
            #np.savetxt("foo.csv", dilated_mask, delimiter=",")



            #sift_image = cv2.drawKeypoints(gray, keypoints, cv_img)
            #cv2.imwrite(f"sift{str(id)}.jpg", sift_image)
            #print(keypoints[0])
            #exit(0)

            #image = Image.open(filename)
            #image_np = np.asarray(image)
            #print(image_np.shape)
            #exit(0)