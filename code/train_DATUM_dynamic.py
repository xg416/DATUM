import argparse
from PIL import Image
import time
import logging
import os
import numpy as np
import random
from datetime import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from utils.scheduler import GradualWarmupScheduler
from TM_model import Model
from TM_model.archs import SpyNet
import utils.losses as losses
from utils.utils_image import eval_tensor_imgs
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint
from data.dataset_LMDB_train import DataLoaderTurbVideo

def get_args():
    parser = argparse.ArgumentParser(description='Train the DATUM on images and restoration')
    parser.add_argument('--iters', type=int, default=800000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=240, help='Batch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='number of iterations to save checkpoint')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='number of iterations for validation')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0002, help='Learning rate', dest='lr')
    parser.add_argument('--num_frames', type=int, default=30, help='number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader')
    parser.add_argument('--train_path', type=str, default='/home/zhan3275/data/ATSyn_LMDB/train_lmdb/', help='path of training imgs')
    parser.add_argument('--train_info', type=str, default='/home/zhan3275/data/ATSyn_LMDB/train_lmdb/train_info.json', help='info of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/ATSyn_LMDB/test_lmdb/', help='path of validation imgs')
    parser.add_argument('--val_info', type=str, default='/home/zhan3275/data/ATSyn_LMDB/test_lmdb/test_info.json', help='info of testing imgs')   
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='/home/zhan3275/Documents/DATUM/train_log', help='path to save logging files and images')
    parser.add_argument('--task', type=str, default='turb', help='choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='DATUM_dynamic', help='name of this running')
    parser.add_argument('--start_over', action='store_true', help='start the scheduler over')

    parser.add_argument('--model', type=str, default='DATUM', help='type of model to construct')
    parser.add_argument('--output_full', action='store_true', help='output # of frames is the same as the input')
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--deform_group', type=int, default=8, help='# of deformable groups')
    parser.add_argument('--deform_type', type=str, default='att', help='type of deformable operation')
    parser.add_argument('--train_flow', action='store_true', help='enable the finetuning of optical flow, true will set spynet_path to None')
    parser.add_argument('--spynet_path', type=str, default="./model_zoo/spynet_init.pth")
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')

    parser.add_argument('--seed', type=int, default=3275, help='random seed')
    parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
    parser.add_argument('--no_profile', action='store_true', help='show # of parameters and computation cost')
    parser.add_argument('--profile_H', type=int, default=720,
                                 help='height of image to generate profile of model')
    parser.add_argument('--profile_W', type=int, default=1280,
                                 help='width of image to generate profile of model')
    return parser.parse_args()

def validate(args, model, val_loader, criterion, iter_count, im_save_freq, im_save_path, device, level):
        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        eval_loss_main = 0
        eval_loss_tilt = 0
        model.eval()
        for s, data in enumerate(val_loader):
            input_ = data[0].to(device)
            tilt = data[1].to(device)
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
                
            with torch.no_grad():
                output, output_noT = model((input_, tilt))
                if not args.output_full:
                    input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
                
                loss1 = criterion(output, target)
                loss2 = criterion(output_noT, target)
                loss = loss1 + loss2 
                eval_loss_main += loss1.item()
                eval_loss_tilt += loss2.item()
            
            if s % im_save_freq == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=im_save_path, kw=level+'val', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            test_results_folder['psnr'] += psnr_batch
            test_results_folder['ssim'] += ssim_batch
                        
        psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
        ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
        eval_loss_main /= (s + 1)
        return psnr, ssim, eval_loss_main
        
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    if not os.path.exists(run_path):
        result_img_path, path_ckpt, path_scipts = create_log_folder(run_path)
    logging.basicConfig(filename=f'{run_path}/recording.log', \
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu_count = torch.cuda.device_count()
    get_cuda_info(logging)
    
    train_dataset = DataLoaderTurbVideo(args.train_path, args.train_info, turb=True, tilt=True, blur=False, \
                                    num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    val_dataset_weak = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='weak', \
                                    num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_weak = DataLoader(dataset=val_dataset_weak, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    val_dataset_medium = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='medium', \
                                    num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_medium = DataLoader(dataset=val_dataset_medium, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    val_dataset_strong = DataLoaderTurbVideo(args.val_path, args.val_info, turb=True, tilt=True, blur=False, level='strong', \
                                    num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader_strong = DataLoader(dataset=val_dataset_strong, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    model = Model(args).cuda()
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
    ######### Scheduler ###########
    total_iters = args.iters
    start_iter = 1
    warmup_iter = 10000
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, total_iters-warmup_iter, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
    scheduler.step()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ######### Resume ###########
    spynet = SpyNet(args.spynet_path).cuda() # if args.spynet_path==None, random initialize
    if args.load:
        if args.load == 'latest':
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f'search for the latest checkpoint of {args.run_name} failed!')
        else:
            load_path = args.load
        checkpoint = torch.load(load_path)

        # if args.spynet_path==None, load from the DATUM checkpoint the same as other weights
        if args.spynet_path is not None: 
            for k,v in spynet.state_dict().items():
                full_k = 'model.spynet.' + k
                checkpoint['state_dict'][full_k].copy_(v)

        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        if not args.start_over:
            if 'epoch' in checkpoint.keys():
                start_iter = checkpoint["epoch"] * len(train_dataset)
            elif 'iter' in checkpoint.keys():
                start_iter = checkpoint["iter"] 
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_lr = optimizer.param_groups[0]['lr']
            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            logging.info(f'==> Resuming Training with learning rate: {new_lr}')
            print('------------------------------------------------------------------------------')
            
        for i in range(1, start_iter):
            scheduler.step()

    if not args.train_flow:
        for name, param in model.named_parameters():
            if 'spynet' in name:
                param.requires_grad = False

    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).cuda()

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss3D()
    
    logging.info(f'''Starting training:
        Total_iters:     {total_iters}
        Start_iters:     {start_iter}
        Batch size:      {args.batch_size}
        Learning rate:   {new_lr}
        Train flow:      {args.train_flow}
        Training size:   {len(train_dataset)}
        val_dataset_weak size: {len(val_dataset_weak)}
        val_dataset_medium size: {len(val_dataset_medium)}
        val_dataset_strong size: {len(val_dataset_strong)}
        Checkpoints:     {path_ckpt}
    ''')
    
    ######### train ###########
    best_psnr = 0
    iter_count = start_iter

    current_start_time = time.time()
    current_loss_main = 0
    current_loss_tilt = 0
    train_results_folder = OrderedDict()
    train_results_folder['psnr'] = []
    train_results_folder['ssim'] = []
    
    model.train()
    for epoch in range(1000000):
        for data in train_loader:
            # zero_grad
            for param in model.parameters():
                param.grad = None
            
            input_ = data[0].to(device)
            tilt = data[1].to(device)
            output, output_noT = model((input_, tilt))
            
            if args.output_full:
                target = data[2].to(device)
            else:
                target = data[2][:, args.past_frames:args.num_frames-args.future_frames, ...].to(device)
                input_ = input_[:, args.past_frames:args.num_frames-args.future_frames, ...]
            loss1 = criterion_char(output, target)
            loss2 = criterion_char(output_noT, target)
            loss = 0.8*loss1 + 0.2*loss2 
            
            # loss = criterion_char(output, target) + 0.05*criterion_edge(output, target)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            optimizer.step()
            scheduler.step()
            current_loss_main += loss1.item()
            current_loss_tilt += loss2.item()
            iter_count += 1

            if iter_count % 500 == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=None, kw='train', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            train_results_folder['psnr'] += psnr_batch
            train_results_folder['ssim'] += ssim_batch

            if iter_count>start_iter and iter_count % args.print_period == 0:
                psnr = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
                ssim = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
                
                logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss1 {:8f} -Loss2 {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}'
                             .format(iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], 
                                     current_loss_main/args.print_period, current_loss_tilt/args.print_period, psnr, ssim))
                
                torch.save({'iter': iter_count, 
                            'psnr': psnr,
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(path_ckpt, f"model_{iter_count}.pth")) 

                torch.save({'iter': iter_count, 
                            'psnr': psnr,
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(path_ckpt, "latest.pth")) 
                current_start_time = time.time()
                current_loss_main = 0
                current_loss_tilt = 0
                train_results_folder = OrderedDict()
                train_results_folder['psnr'] = []
                train_results_folder['ssim'] = []
                                          
            #### Evaluation ####
            if iter_count>0 and iter_count % args.val_period == 0:
                psnr_w, ssim_w, loss_w = validate(args, model, val_loader_weak, criterion_char, iter_count, 200, result_img_path, device, 'weak')
                logging.info('Validation W: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, loss_w, psnr_w, ssim_w))
                
                psnr_m, ssim_m, loss_m = validate(args, model, val_loader_medium, criterion_char, iter_count, 200, result_img_path, device, 'medium')
                logging.info('Validation M: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, loss_m, psnr_m, ssim_m))
                
                psnr_s, ssim_s, loss_s = validate(args, model, val_loader_strong, criterion_char, iter_count, 200, result_img_path, device, 'strong')
                logging.info('Validation S: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, loss_s, psnr_s, ssim_s))
                psnr = (psnr_w + psnr_m + psnr_s) / 3
                ssim = (ssim_w + ssim_m + ssim_s) / 3
                val_loss = (loss_w + loss_m + loss_s) / 3
                logging.info('Validation All: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, val_loss, psnr, ssim))
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({'iter': iter_count,
                                'psnr': psnr,
                                'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(path_ckpt, "model_best.pth"))
                model.train()
                
if __name__ == '__main__':
    main()
