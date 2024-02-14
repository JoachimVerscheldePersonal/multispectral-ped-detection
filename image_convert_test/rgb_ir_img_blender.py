#!/usr/bin/python3

import os
import glob
import tqdm
import argparse
import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

def blend_rgb_ir(img_rgb: NDArray, img_ir: NDArray) -> (NDArray):

    # convert ir and rgb images to LAB colorspace
    img_rgb_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_ir_lab = cv2.cvtColor(img_ir, cv2.COLOR_RGB2LAB)

    # split the LAB images into l, a and b channels
    l_rgb, a_rgb, b_rgb  = cv2.split(img_rgb_lab)
    l_ir, _, _ = cv2.split(img_ir_lab)
    
    # normalize L channels to 0-255
    if np.max(l_rgb) < 250:
        l_rgb = cv2.normalize(l_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if np.max(l_ir) < 250:
        l_ir = cv2.normalize(l_ir, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # do processing on the L channels
    # examples:
    #       processed_channel = (abs(l_rgb - l_ir)).astype(np.uint8)
    #       processed_channel = ((l_rgb + l_ir)/2).astype(np.uint8)
    processed_channel = np.maximum(l_rgb, l_ir)

    # normalize the processed channel to 0-255 (not always necessary, depends on the processing)
    #processed_channel = cv2.normalize(processed_channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # merge the processed channel with the other channels and transform back to RGB
    result_image = cv2.merge((processed_channel, a_rgb, b_rgb))
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_LAB2RGB)

    return result_image_rgb

image_vis_lst = []
image_ir_lst = []

# root directory for image scanning is mandatory
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help = "Root directory for images", required = True)
args = parser.parse_args()

base_path = r'{}/**/visible/*.jpg'.format(args.directory)
print("Scanning for images at {}".format(base_path))

for vis_file in glob.glob(base_path, recursive=True):
    image_vis_lst.append(vis_file)
    ir_file = vis_file.replace("visible", "lwir")
    if os.path.exists(ir_file):
        image_ir_lst.append(ir_file)

if len(image_vis_lst) != len(image_ir_lst):
    print("Visible and lwir image lists do not match, exiting!")
    print(len(image_vis_lst))
    print(len(image_ir_lst))
    exit(1)

if len(image_vis_lst) == 0:
    print("No images to process, exiting!")
    exit(1)

print("Scan complete, converting and copying images...")

#for idx in range(len(image_vis_lst)):
for idx in tqdm.tqdm(range(len(image_vis_lst))):
    merge_file = image_vis_lst[idx].replace("visible", "blended")
    filen = os.path.basename(image_vis_lst[idx])

    if os.path.exists(merge_file):
        continue

    output_dir = os.path.dirname(merge_file)
    if os.path.isdir(output_dir) != True:
        os.makedirs(output_dir)

    #print("processing image {}/{}: {}".format(idx+1, len(image_vis_lst), image_vis_lst[idx]))

    img_rgb = cv2.imread(image_vis_lst[idx])
    img_ir = cv2.imread(image_ir_lst[idx])
    img_out  = blend_rgb_ir(img_rgb, img_ir)

    ret = cv2.imwrite(merge_file, img_out)
    if ret != True:
        print("Could not write blended image, exiting!")
        exit(1)
      
print("Conversion complete.")
