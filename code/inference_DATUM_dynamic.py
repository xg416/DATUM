import argparse
from PIL import Image

import os
import cv2
import numpy as np
import torch
from TM_model import Model
import torchvision.transforms.functional as TF

def get_args():
    parser = argparse.ArgumentParser(description='test the DATUM on images and restoration') 
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--input_path', type=str, default=None, help='path of input video')
    parser.add_argument('--out_path', type=str, default=None, help='path of output video')
    parser.add_argument('--start_frame', type=int, default=0, help='start index of frames')
    parser.add_argument('--num_frames', type=int, default=6000, help='max number of frames')
    parser.add_argument('--clip', type=int, default=60, help='max number of frames in a temporal patch')
    parser.add_argument('--ps', type=int, default=2000, help='max spatial dimensions for a patch')
    parser.add_argument('--model', type=str, default='DATUM', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default=None)
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    return parser.parse_args()

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
    
def temp_segment(total_frames, chunk_len, valid_len):
    residual = (chunk_len - valid_len) // 2
    test_frame_info = [{'start':0, 'range':[0,chunk_len-residual]}]
    num_chunk = (total_frames-1) // valid_len
    for i in range(1, num_chunk+1):
        if i == num_chunk:
            test_frame_info.append({'start':total_frames-chunk_len, 'range':[i*valid_len+residual,total_frames]})
        elif i*valid_len+chunk_len >= total_frames:
            test_frame_info.append({'start':total_frames-chunk_len, 'range':[i*valid_len+residual, total_frames]})
            break
        else:
            test_frame_info.append({'start':i*valid_len, 'range':[i*valid_len+residual,i*valid_len+chunk_len-residual]})
    return len(test_frame_info), test_frame_info

def tensor2img(tensor, fidx):
    img = tensor[0, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img   
    
    
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(args).cuda()
checkpoint = torch.load(args.load)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)

max_frames = args.num_frames
start_frame = args.start_frame
resize = 1
# 'G00393_set1_rand_1622563286641_bd4061b9'
input_path = args.input_path
output_path = args.out_path
# clip_path = './results/video_19_in.mp4'
with torch.no_grad():
    turb_vid = cv2.VideoCapture(input_path)
    h, w = int(turb_vid.get(4)), int(turb_vid.get(3))
    fps = int(turb_vid.get(5))
    total_frames = int(turb_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if total_frames>max_frames:
        total_frames = max_frames
    all_frames = [turb_vid.read()[1] for i in range(total_frames)]
    all_frames = [f for f in all_frames if f is not None]
    
    # output_writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
    # for frame in all_frames:
    #     output_writer.write(frame)
    # output_writer.release()
    
    if resize<1:
        w = int(w*resize)
        h = int(h*resize)
        all_frames = [cv2.resize(f, (w, h)) for f in all_frames]
    total_frames = len(all_frames)
    print(f"video {input_path}, input frames {total_frames}, h {h} x w {w}")
    cl = args.clip
    if total_frames > cl:
        n_chunks, test_frame_info = temp_segment(total_frames, chunk_len=cl, valid_len=cl-4)
    else:
        n_chunks, test_frame_info = temp_segment(total_frames, chunk_len=total_frames, valid_len=total_frames)
        cl = total_frames
        
    out_frames = []
    frame_idx = 0
    patch_unit = 8
    if h%patch_unit==0:
        nh = h
    else:
        nh = h//patch_unit*patch_unit + patch_unit
    if w%patch_unit==0:
        nw = w
    else:
        nw = w//patch_unit*patch_unit + patch_unit 
    padw, padh = nw-w, nh-h

    for i in range(n_chunks):
        out_range = test_frame_info[i]['range']
        in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+cl]
        print(out_range,in_range)
        inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in all_frames[in_range[0]:in_range[1]]]
        inp_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in inp_imgs]
        inp_imgs = [TF.to_tensor(img) for img in inp_imgs]
        input_ = torch.stack(inp_imgs, dim=0).unsqueeze(0).cuda()
        if max(h,w)>args.ps:
            output = test_spatial_overlap(input_, model, args.ps, reduce_frame=4)
        else:
            output,_ = model((input_, input_))
        if i==0:
            for j in range(output.shape[1]):
                out = cv2.cvtColor(tensor2img(output, j), cv2.COLOR_RGB2BGR)
                out_frames.append(out)
        else:           
            for j in range(out_range[0]-2-in_range[0], output.shape[1]):
                out = cv2.cvtColor(tensor2img(output, j), cv2.COLOR_RGB2BGR)
                out_frames.append(out)
        torch.cuda.empty_cache()
            
    print(f"video {input_path} done! input frames {total_frames}, output frames {len(out_frames)}")
    output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
    for fid, frame in enumerate(out_frames):
        output_writer.write(frame[:h, :w, :])
        #output_writer.write(cv2.resize(frame, (h*2,w*2)))
    output_writer.release()
