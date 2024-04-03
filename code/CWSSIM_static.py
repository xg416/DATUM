import time
import os

from PIL import Image

from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

save_dir = '/home/zhan3275/data/test_DATUM_static'
log_path = '/home/zhan3275/data/test_DATUM_static/1CWSSIM.txt'
cw_ssim_list = []
cw_ssim_img = []
for name in os.listdir(save_dir):
    if name.startswith('Places'):
        start = time.time()
        im_dir = os.path.join(save_dir, name)
        im_gt = Image.open(os.path.join(im_dir, 'gt.png'))
        try:
            for j in range(4):
                im_restore = Image.open(f'{im_dir}/out_{j}.png')
                cw_ssim = SSIM(im_gt).cw_ssim_value(im_restore)
                cw_ssim_img.append(cw_ssim)
        except:
            print(f"!!!!Problem in {im_dir}")

        cw = sum(cw_ssim_img) / len(cw_ssim_img)
        cw_ssim_list.append(cw)
        all_time = time.time()-start
        cw_ssim_img = []
        print(f'image {name} finished, CW-SSIM score is {cw}, use time {all_time}')
        with open(log_path, 'a') as file:
            file.write(f'{name}, {cw} \n')

# for i in range(1000):
#     start = time.time()
#     im_dir = os.path.join(save_dir, str(i))
#     im_gt = Image.open(os.path.join(im_dir, 'gt.png'))
#     for j in range(23,27):
#         im_restore = Image.open(f'{im_dir}/out/{j}.png')
#         cw_ssim = SSIM(im_gt).cw_ssim_value(im_restore)
#         cw_ssim_img.append(cw_ssim)

#     cw = sum(cw_ssim_img) / len(cw_ssim_img)
#     cw_ssim_list.append(cw)
#     all_time = time.time()-start
#     cw_ssim_img = []
#     print(f'image {j} finished, CW-SSIM score is {cw}, use time {all_time}')
#     with open(log_path, 'a') as file:
#         file.write(f'{j}, {cw} \n')
            
cw_final = sum(cw_ssim_list) / len(cw_ssim_list)
with open(log_path, 'a') as file:
    file.write(f'final CW-SSIM is {cw_final}')
print(f'final CW-SSIM is {cw_final}')
