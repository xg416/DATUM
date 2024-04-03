import argparse
import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from TM_model import Model
import cv2
from utils.utils_image import test_tensor_img
from data.dataset_LMDB_train import DataLoaderTurbVideoTest
import json, math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def get_args():
    parser = argparse.ArgumentParser(description='Test the DATUM on images and restoration')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=512, help='Patch size')
    parser.add_argument('--num_frames', type=int, default=120, help='max number of frames')
    parser.add_argument('--chunk_frames', type=int, default=120, help='frames in split chunks')
    parser.add_argument('--data_path', type=str, default='/home/zhan3275/data/ATSyn_LMDB/test_lmdb/', help='path of validation imgs')
    parser.add_argument('--info_path', type=str, default='/home/zhan3275/data/ATSyn_LMDB/test_lmdb/test_info.json', help='info of testing imgs')  
    parser.add_argument('--result_path', '-result', type=str, default='/home/zhan3275/data/test_DATUM_video', help='path of validation imgs')
    parser.add_argument('--model_path', '-mp', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader')
    
    parser.add_argument('--model', type=str, default='DATUM', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default=None)
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    
    parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
    parser.add_argument('--no_profile', action='store_true', help='show # of parameters and computation cost')
    parser.add_argument('--profile_H', type=int, default=720,
                                 help='height of image to generate profile of model')
    parser.add_argument('--profile_W', type=int, default=1280,
                                 help='width of image to generate profile of model')
    return parser.parse_args()


def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = img * 255.0  # float32 
    return img

def split_to_patches(h, w, s):
    nh = h // s + 1
    nw = w // s + 1
    if nh > 1:
        ol_h = int((nh * s - h) / (nh - 1))
        h_start = 0
        hpos = [h_start]
        for i in range(1, nh):
            h_start = hpos[-1] + s - ol_h
            if h_start+s > h:
                h_start = h-s
            hpos.append(h_start)      
        if len(hpos)==2 and hpos[0] == hpos[1]:
            hpos = [hpos[0]]
    else:
        hpos = [0]
    if nw > 1:
        ol_w = int((nw * s - w) / (nw - 1))
        w_start = 0  
        wpos = [w_start]
        for i in range(1, nw):
            w_start = wpos[-1] + s - ol_w
            if w_start+s > w:
                w_start = w-s
            wpos.append(w_start)
        if len(wpos)==2 and wpos[0] == wpos[1]:
            wpos = [wpos[0]]
    else:
        wpos = [0]
    return hpos, wpos
    
def test_spatial_overlap(input_blk, model, patch_size, reduce_frame=0):
    _,l,c,h,w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros(_,l-reduce_frame,c,h,w).cuda()
    out_masks = torch.zeros(_,l-reduce_frame,c, h,w).cuda()
    for hi in hpos:
        for wi in wpos:
            if h > patch_size:
                h_end = hi+patch_size
            else:
                h_end = h
            if w > patch_size:
                w_end = wi+patch_size
            else:
                w_end = w       
            input_ = input_blk[..., hi:h_end, wi:w_end]
            output_, _ = model((input_, input_))
            out_spaces[..., hi:h_end, wi:w_end].add_(output_)
            out_masks[..., hi:h_end, wi:w_end].add_(torch.ones_like(output_))
    return out_spaces / out_masks
    

def temp_segment_rnn(total_frames, input_len, output_len, left_pad_frames=2, right_pad_frames=2):
    if total_frames < input_len:
        return [0, total_frames], [left_pad_frames, total_frames-right_pad_frames]
    n_chunks = math.ceil((total_frames-left_pad_frames-right_pad_frames) / output_len)
    in_range = []
    out_range = []
    last_out_e = 0
    for i in range(n_chunks):
        out_s = left_pad_frames + i * output_len # 2, 12, 22, 32
        out_e = out_s + output_len
        
        in_e = out_e + 2
        in_s = out_s - 2

        if in_e > total_frames:
            in_e = total_frames
            in_s = in_e - input_len
            out_e = in_e - 2
        out_range.append([out_s, out_e])
        in_range.append([in_s, in_e])
    return in_range, out_range

     
def test(args, model, dataloader, result_dir):
    patch_size = args.patch_size
    chunk_frames = args.chunk_frames
    input_len = chunk_frames + 2
    # output_len = chunk_frames - 2
    b = 0
    fps = 30.0
    psnr = []
    ssim = []
    tmf_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
    lpips = []
    PSNR_all = {}
    with torch.no_grad():
        for data in dataloader:
            input_all = data[0].cuda()
            gt = data[1].cuda()
            h, w = data[2][b].item(), data[3][b].item()
            total_frames = data[4][b].item()
            path = os.path.split(data[-1][b])[-1]
            total_frames = input_all.shape[1]
            print(total_frames)
            if input_len > total_frames:
                input_len = total_frames

            img_result_path = os.path.join(result_dir, path.split('.')[0]+'.mp4')
            if os.path.exists(img_result_path):
                continue
            in_ranges, out_ranges = temp_segment_rnn(total_frames, input_len, input_len-4)
            output = torch.empty_like(input_all)
            for in_range, out_range in zip(in_ranges, out_ranges):
                print(in_range, out_range)
                input_ = input_all[:, in_range[0]:in_range[1], ...]
                if max(w,h) <= patch_size:
                    output[:, in_range[0]+2:out_range[1], ...], _ = model((input_, input_))
                else:   
                    output[:, in_range[0]+2:out_range[1], ...] = test_spatial_overlap(input_, model, patch_size, args.past_frames+args.future_frames)
            
            output = output[:, args.past_frames:total_frames-args.future_frames, :, :h, :w]
            gt = gt[:, args.past_frames:total_frames-args.future_frames, :, :h, :w]
            out_frames, psnr_video, ssim_video, lpips_video = test_tensor_img(gt, output, tmf_lpips)
    
            with open(f'{result_dir}/result.log', 'a') as log_file:
                log_file.write(f'video:{path}, psnr:{sum(psnr_video)/len(psnr_video)}, ssim:{sum(ssim_video)/len(ssim_video)}, LPIPS:{sum(lpips_video)/len(lpips_video)}\n')

            psnr.append(sum(psnr_video)/len(psnr_video))
            ssim.append(sum(ssim_video)/len(ssim_video))
            lpips.append(sum(lpips_video)/len(lpips_video))
            PSNR_all[path] = psnr_video
            output_writer = cv2.VideoWriter(img_result_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w,h))
            for fid, frame in enumerate(out_frames):
                output_writer.write(frame)
            output_writer.release()
    return psnr, ssim, lpips, PSNR_all

def main():
    args = get_args()
    input_dir = args.data_path
    result_dir = args.result_path
    model_path = args.model_path
    patch_size = args.patch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        
    #logging.basicConfig(filename=f'{result_dir}/result.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    log_file = open(f'{result_dir}/result.log', 'a')
    test_dataset_weak = DataLoaderTurbVideoTest(args.data_path, args.info_path, turb=True, tilt=False, blur=False, level='weak', \
                                    num_frames=args.num_frames, patch_unit=8)
    test_loader_weak = DataLoader(dataset=test_dataset_weak, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    test_dataset_medium = DataLoaderTurbVideoTest(args.data_path, args.info_path, turb=True, tilt=False, blur=False, level='medium', \
                                    num_frames=args.num_frames, patch_unit=8)
    test_loader_medium = DataLoader(dataset=test_dataset_medium, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    test_dataset_strong = DataLoaderTurbVideoTest(args.data_path, args.info_path, turb=True, tilt=False, blur=False, level='strong', \
                                    num_frames=args.num_frames, patch_unit=8)
    test_loader_strong = DataLoader(dataset=test_dataset_strong, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    model = Model(args).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    model.eval()

    psnr = []
    ssim = []
    lpips = []
    PSNR_all = {}
    
    psnr_weak, ssim_weak, lpips_weak, PSNR_pframe_weak = test(args, model, test_loader_weak, result_dir)
    psnr += psnr_weak
    ssim += ssim_weak
    lpips += lpips_weak
    PSNR_all.update(PSNR_pframe_weak)

    psnr_medium, ssim_medium, lpips_medium, PSNR_pframe_medium = test(args, model, test_loader_medium, result_dir)
    psnr += psnr_medium
    ssim += ssim_medium
    lpips += lpips_medium
    PSNR_all.update(PSNR_pframe_medium)

    psnr_strong, ssim_strong, lpips_strong, PSNR_pframe_strong = test(args, model, test_loader_strong, result_dir)
    psnr += psnr_strong
    ssim += ssim_strong
    lpips += lpips_strong
    PSNR_all.update(PSNR_pframe_strong)
    
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Weak turb psnr:{sum(psnr_weak)/len(psnr_weak)}, ssim:{sum(ssim_weak)/len(ssim_weak)}, lpips:{sum(lpips_weak)/len(lpips_weak)}\n')
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Medium turb psnr:{sum(psnr_medium)/len(psnr_medium)}, ssim:{sum(ssim_medium)/len(ssim_medium)}, lpips:{sum(lpips_medium)/len(lpips_medium)}\n') 
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Strong turb psnr:{sum(psnr_strong)/len(psnr_strong)}, ssim:{sum(ssim_strong)/len(ssim_strong)}, lpips:{sum(lpips_strong)/len(lpips_strong)}\n')
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Overall psnr:{sum(psnr)/len(psnr)}, ssim:{sum(ssim)/len(ssim)}, lpips:{sum(lpips)/len(lpips)}')
        
    with open(f'{result_dir}/PSNR_pframe.json', 'w') as jsonfile:
        json.dump(PSNR_all, jsonfile)

if __name__ == '__main__':
    main()
