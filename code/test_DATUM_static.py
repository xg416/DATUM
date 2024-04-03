import argparse
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from TM_model import Model
from data.dataset_video_train import DataLoaderTurbImageTest
from utils.utils_image import test_tensor_img
import json

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=512, help='Patch size')
    parser.add_argument('--loaded_frames', type=int, default=20, help='number of frames for the model')
    parser.add_argument('--valid_frames', type=int, default=4, help='number of frames for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/ATSyn_static/test_static', help='path of validation imgs')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--result_path', '-result', type=str, default='/home/zhan3275/data/test_DATUM_static', help='path of validation imgs')
    
    parser.add_argument('--model', type=str, default='DATUM', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default=None)
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    return parser.parse_args()

def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
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

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_size = args.patch_size
    num_frames = args.loaded_frames
    result_dir = args.result_path
    os.makedirs(result_dir, exist_ok=True)
    log_file = open(f'{result_dir}/result.log', 'a')
    
    start_frame = (50 - num_frames) // 2
    valid_frames = args.valid_frames
    valid_start_frame = num_frames - valid_frames - 2
    
    test_dataset = DataLoaderTurbImageTest(rgb_dir=args.val_path, num_frames=num_frames, total_frames=50, \
        im_size=args.patch_size, noise=0.0001, other_mod='tilt', start_frame=start_frame)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    
    model = Model(args).cuda()
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    model.eval()

    PSNR_all = {}
    with torch.no_grad():
        psnr = []
        ssim = []
        for s, data in enumerate(test_loader):
            input_ = data[1].to(device)
            target = data[2].to(device)
            target = target.unsqueeze(1).repeat(1, valid_frames, 1, 1, 1)
            seq_path = os.path.split(data[3][0])[-1]
            print(seq_path, target.shape, input_.shape)
            out_folder = os.path.join(result_dir, seq_path)
            os.makedirs(out_folder, exist_ok=True)
            h, w = input_.shape[-2], input_.shape[-1]
            output = torch.empty_like(input_)
            if max(w,h) <= patch_size:
                output[:, 2:-2, ...], _ = model((input_, input_))
            else:   
                output[:, 2:-2, ...] = test_spatial_overlap(input_, model, patch_size, args.past_frames+args.future_frames)
            output = output[:, valid_start_frame:valid_start_frame+valid_frames, ...]
            input_ = input_[:, valid_start_frame:valid_start_frame+valid_frames, ...]

            out_frames, psnr_frames, ssim_frames = test_tensor_img(target, output)
            
            with open(f'{result_dir}/result.log', 'a') as log_file:
                log_file.write(f'sequence:{seq_path}, psnr:{sum(psnr_frames)/len(psnr_frames)}, ssim:{sum(ssim_frames)/len(ssim_frames)}\n')
            psnr.append(sum(psnr_frames)/len(psnr_frames))
            ssim.append(sum(ssim_frames)/len(ssim_frames))
            PSNR_all[seq_path] = psnr_frames
            for i in range(valid_frames):
                # in_image = restore_PIL(input_, 0, i)
                # in_save = Image.fromarray(in_image)
                # in_save.save(os.path.join(out_folder, f'in_{i}.png'), "PNG")
                out_save = Image.fromarray(out_frames[i])
                out_save.save(os.path.join(out_folder, f'out_{i}.png'), "PNG")

            gt_image = restore_PIL(target, 0, 0)
            gt_save = Image.fromarray(gt_image)
            gt_save.save(os.path.join(out_folder, f'gt.png'), "PNG")
            
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Overall psnr:{sum(psnr)/len(psnr)}, ssim:{sum(ssim)/len(ssim)}\n')
    with open(f'{result_dir}/PSNR_pframe.json', 'w') as jsonfile:
        json.dump(PSNR_all, jsonfile)
      
if __name__ == '__main__':
    main()
