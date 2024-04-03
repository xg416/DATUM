import argparse
from PIL import Image
import os
import cv2
import numpy as np
import torch
from TM_model import Model
import torchvision.transforms.functional as TF

def read_images(path, prefix, start_id=1, length=100, resize=1):
    imgs = []
    for i in range(start_id, length+start_id):
        img_path = os.path.join(path, prefix.format(i))
        img = cv2.imread(img_path)
        if resize != 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*resize), int(h*resize)))
        imgs.append(img)
    return imgs

def tensor2img(tensor, fidx):
    img = tensor[0, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img   

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration') 
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--n_frames', type=int, default=60, help='base # of channels for Conv')
    parser.add_argument('--resize', type=int, default=360, help='base # of channels for Conv')
  
    parser.add_argument('--model', type=str, default='DATUM', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default=None)
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    return parser.parse_args()

    
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(args).cuda()
checkpoint = torch.load(args.load)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)

input_dir = '/home/zhan3275/data/Turb_text/'
output_dir = f'/home/zhan3275/data/DATUM/text/f_{args.n_frames}_{args.resize}'
start_frame = 1 + (100-args.n_frames)//2
resize = float(args.resize)/440
# input_dir = '/home/zhan3275/data/OTIS/Color/Fixed Patterns'
# output_dir = '/home/zhan3275/data/OTIS/OTIS_result'
with torch.no_grad():
    for v in os.listdir(input_dir)[::-1]:
        input_path = os.path.join(input_dir, v)
        output_path = os.path.join(output_dir, v)
        os.makedirs(output_path, exist_ok=True)
        print(f"{input_path}")
        
        all_frames = read_images(input_path, "data_{:04d}.png", start_id=start_frame, length=args.n_frames, resize=resize) # 256:0.581
        # print(all_frames[0].shape)
        h, w = all_frames[0].shape[:2]
        total_frames = len(all_frames)
            
        out_frames = []
        frame_idx = 0
        patch_unit = 4
        if h%patch_unit==0:
            nh = h
        else:
            nh = h//patch_unit*patch_unit + patch_unit
        if w%patch_unit==0:
            nw = w
        else:
            nw = w//patch_unit*patch_unit + patch_unit 
        padw, padh = nw-w, nh-h

        inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in all_frames]
        inp_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in inp_imgs]
        inp_imgs = [TF.to_tensor(img) for img in inp_imgs]
        input_ = torch.stack(inp_imgs, dim=0).unsqueeze(0).cuda()
        # output = model(input_)
        output, _ = model((input_, input_))
        num_out = len(inp_imgs)-4
        for j in range(num_out//2-2,num_out//2+2):
            out = cv2.cvtColor(tensor2img(output, j), cv2.COLOR_RGB2BGR)
            out_frames.append(out)
        torch.cuda.empty_cache()

        print(f"{output_path} done! input frames {total_frames}, output frames {len(out_frames)}")
        for fid, frame in enumerate(out_frames):
            result_frame_path = os.path.join(output_path, "result_{:04d}.png".format(fid+1))
            cv2.imwrite(result_frame_path, cv2.resize(frame, (440, 440)))
