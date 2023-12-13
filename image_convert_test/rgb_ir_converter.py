#!/usr/bin/python3

import os
import glob
import tempfile
import sys
import tqdm

image_vis_lst = []
image_ir_lst = []

base_path = r'{}/**/visible/*.jpg'.format(sys.argv[1])
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

with tempfile.TemporaryDirectory() as tmpdir:
    print('Created temporary directory', tmpdir)
    #for idx in range(len(image_vis_lst)):
    for idx in tqdm.tqdm(range(len(image_vis_lst))):
        merge_file = image_vis_lst[idx].replace("visible", "merged")
        filen = os.path.basename(image_vis_lst[idx])

        #print("processing image {}/{}: {}".format(idx+1, len(image_vis_lst), image_vis_lst[idx]))

        cmd_rgb = "/usr/bin/convert {} -colorspace Lab -separate {}/rgb.png".format(image_vis_lst[idx], tmpdir)
        cmd_ir = "/usr/bin/convert {} -colorspace Lab -separate {}/thermal.png".format(image_ir_lst[idx], tmpdir)
        cmd_merge = "/usr/bin/convert {}/thermal-0.png {}/rgb-0.png {}/rgb-2.png  -normalize -set colorspace Lab -combine {}/{}".format(tmpdir, tmpdir, tmpdir, tmpdir, filen) 
        cmd_copy = "/usr/bin/install -D {}/{} {}".format(tmpdir, filen, merge_file)
        cmd_rm = "/usr/bin/rm {}/{}".format(tmpdir, filen)
        
        cmd_lst = [cmd_rgb, cmd_ir, cmd_merge, cmd_copy, cmd_rm]
        for cmd in cmd_lst:
            ret = os.system(cmd)
            if ret != 0:
                print("Error while processing, exiting!")
                exit(1)

print("Conversion complete.")
