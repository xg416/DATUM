import time
import os
import cv2
import json
import numpy as np
from PIL import Image
import argparse
from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

def get_args():
    parser = argparse.ArgumentParser(description='output video dir')
    parser.add_argument('--save_dir', type=str, default='/home/zhan3275/data/test_DATUM_video/')
    parser.add_argument('--skip_frames', type=int, default=2, help='skip first n frames of gt')
    return parser.parse_args()

args = get_args()

save_dir = args.save_dir
skip_frames = args.skip_frames
log_path = os.path.join(save_dir, "1CWSSIM.txt")
gt_dir = '/home/zhan3275/data/ATSyn_dynamic/test/gt/'
param_dir = '/home/zhan3275/data/ATSyn_dynamic/test/turb_param/'
cw_ssim_list = []
cw_ssim_img = []

cw_ssim_level = {'weak':[], 'medium':[], 'strong':[]}

for v_name in os.listdir(gt_dir):
    start = time.time()
    print(v_name)
    gt_path = os.path.join(gt_dir, v_name)
    save_path = os.path.join(save_dir, v_name)
    param_path = os.path.join(param_dir, v_name.replace(".mp4", ".json"))
    param = json.load(open(param_path, "r"))
    level = param["level"]

    cap_gt = cv2.VideoCapture(gt_path)
    cap_result = cv2.VideoCapture(save_path)
    
    ret_gt, frame_gt = cap_gt.read()
    for i in range(skip_frames):
        ret_gt, frame_gt = cap_gt.read()
    ret_r, frame_r = cap_result.read()
    while ret_gt and ret_r:
        im_gt = Image.fromarray((255*frame_gt).astype(np.uint8))
        im_r = Image.fromarray((255*frame_r).astype(np.uint8))
        cw_ssim = SSIM(im_gt).cw_ssim_value(im_r)
        cw_ssim_img.append(cw_ssim)
        ret_gt, frame_gt = cap_gt.read()
        ret_r, frame_r = cap_result.read()    
    cap_gt.release()
    cap_result.release()
    
    cw = sum(cw_ssim_img) / len(cw_ssim_img)
    cw_ssim_list.append(cw)
    all_time = time.time()-start
    cw_ssim_img = []
    with open(log_path, 'a') as file:
        file.write(f'{v_name}: {cw} \n')
    print(f'{v_name} finished, {level}, CW-SSIM {cw}, time {all_time}')
    cw_ssim_level[level].append(cw)

cw_final = sum(cw_ssim_list) / len(cw_ssim_list)
with open(log_path, 'a') as file:
    file.write(f'weak CW-SSIM is {sum(cw_ssim_level["weak"]) / len(cw_ssim_level["weak"])}\n')
    file.write(f'medium CW-SSIM is {sum(cw_ssim_level["medium"]) / len(cw_ssim_level["medium"])}\n')
    file.write(f'strong CW-SSIM is {sum(cw_ssim_level["strong"]) / len(cw_ssim_level["strong"])}\n')
    file.write(f'final CW-SSIM is {cw_final}')
print(f'final CW-SSIM is {cw_final}')

