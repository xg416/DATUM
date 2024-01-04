from logging import root
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import lmdb
import json
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTurbVideo(Dataset):
    def __init__(self, data_path, info_path, turb=True, tilt=False, blur=False, blur2=False, 
                        level='all', num_frames=12, patch_size=None, noise=None, is_train=True):
        super(DataLoaderTurbVideo, self).__init__()
        self.num_frames = num_frames
        self.require_turb = turb
        self.require_tilt = tilt
        self.require_blur = (blur or blur2)

        if is_train:
            # Can only choose blur or blur2, not together
            if blur2:
                self.blur_dir = os.path.join(data_path, 'train_blur2')
            else:
                self.blur_dir = os.path.join(data_path, 'train_blur')
            self.turb_dir = os.path.join(data_path, 'train_turb')
            self.tilt_dir = os.path.join(data_path, 'train_tilt')
            self.gt_dir = os.path.join(data_path, 'train_gt')
        else:
            if blur2:
                self.blur_dir = os.path.join(data_path, 'test_blur2')
            else:
                self.blur_dir = os.path.join(data_path, 'test_blur')
            self.turb_dir = os.path.join(data_path, 'test_turb')
            self.tilt_dir = os.path.join(data_path, 'test_tilt')
            self.gt_dir = os.path.join(data_path, 'test_gt')
            
        seqs_info = json.load(open(info_path, 'r'))
        
        if level == 'all':
            self.seqs_info = {}
            count = 0
            for info in seqs_info.values():
                if type(info)==dict:
                    self.seqs_info[count] = info
                    count += 1
        else:
            self.seqs_info = {}
            count = 0
            for info in seqs_info.values():
                if type(info)==dict and info['turb_level'] == level:
                    self.seqs_info[count] = info
                    count += 1
                    
        self.sizex = len(self.seqs_info)
        self.env = {}
        self.txn = {}
        self.env['gt'] = lmdb.open(self.gt_dir, readonly=True, lock=False, readahead=True, map_size=1099511627776)
        self.txn['gt'] = self.env['gt'].begin()
        self.required = ['gt']
        if self.require_turb:
            self.env['turb'] = lmdb.open(self.turb_dir, readonly=True, lock=False, readahead=True, map_size=1099511627776)
            self.txn['turb'] = self.env['turb'].begin()
            self.required.append('turb')
        if self.require_tilt:
            self.env['tilt'] = lmdb.open(self.tilt_dir, readonly=True, lock=False, readahead=True, map_size=1099511627776)
            self.txn['tilt'] = self.env['tilt'].begin()
            self.required.append('tilt')
        if self.require_blur:
            self.env['blur'] = lmdb.open(self.blur_dir, readonly=True, lock=False, readahead=True, map_size=1099511627776)
            self.txn['blur'] = self.env['blur'].begin()
            self.required.append('blur')
        
        self.ps = patch_size
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5)*torch.randn(img.shape)
        out = img + noise
        return out.clamp(0,1)
        
    def _fetch_chunk_val(self, idx):
        ps = self.ps
        info = self.seqs_info[idx]
        vname = info['video_name']
        h, w, total_frames = info['h'], info['w'], info['length'] 
        c = 3
        if total_frames < self.num_frames:
            print('no enough frame in video ' + self.gt_list[idx])
        start_frame_id = random.randint(0, total_frames-self.num_frames)
        
        # load frames from database
        imgs = {}
        for modality in self.required:
            imgs[modality] = []
        for fi in range(start_frame_id, start_frame_id+self.num_frames):
            key = '{:s}_{:05d}'.format(vname, fi)
            for modality in self.required:
                enc_key = self.txn[modality].get(key.encode())
                load_img = np.frombuffer(enc_key, dtype='uint8')
                imgs[modality].append(load_img.reshape(h,w,c))
        
        for modality in self.required:
            imgs[modality] = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs[modality]]   
        
        if ps > 0:
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
            for modality in self.required:
                if padw!=0 or padh!=0:
                    imgs[modality] = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in imgs[modality]]   
                imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]

            hh, ww = imgs['gt'][0].shape[1], imgs['gt'][0].shape[2]
            rr, cc = (hh-ps) // 2, (ww-ps) // 2
            # Crop patch
            for modality in self.required:
                imgs[modality] = [img[:, rr:rr+ps, cc:cc+ps] for img in imgs[modality]]
        else:
            for modality in self.required:
                imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]
        
        if self.noise:
            noise_level = self.noise * random.random()
            for modality in self.required:
                if modality == 'blur' or modality == 'turb':
                    imgs[modality] = [self._inject_noise(img, noise_level) for img in imgs[modality]]  
        return imgs
                         
    def _fetch_chunk_train(self, idx):
        ps = self.ps
        info = self.seqs_info[idx]
        vname = info['video_name']
        h, w, total_frames = info['h'], info['w'], info['length'] 
        c = 3
        if total_frames < self.num_frames:
            print('no enough frame in video ' + self.gt_list[idx])
        start_frame_id = random.randint(0, total_frames-self.num_frames)
        
        # load frames from database
        imgs = {}
        for modality in self.required:
            imgs[modality] = []
        for fi in range(start_frame_id, start_frame_id+self.num_frames):
            key = '{:s}_{:05d}'.format(vname, fi)
            for modality in self.required:
                enc_key = self.txn[modality].get(key.encode())
                load_img = np.frombuffer(enc_key, dtype='uint8')
                imgs[modality].append(load_img.reshape(h,w,c))

        if imgs['gt'][0] is None:
            print(self.gt_list[idx], "has no gt image!")

        for modality in self.required:
            imgs[modality] = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs[modality]]
     
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            for modality in self.required:
                imgs[modality] = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in imgs[modality]]
            
        aug    = random.randint(0, 2)
        if aug == 1:
            for modality in self.required:
                imgs[modality] = [TF.adjust_gamma(img, 1) for img in imgs[modality]]
            
        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            for modality in self.required:
                imgs[modality] = [TF.adjust_saturation(img, sat_factor) for img in imgs[modality]]   
            
        hh, ww = h, w
        enlarge_factor = random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        hcro = int(crop_size * random.uniform(1.1, 0.9))
        wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = min(hcro, hh)
        wcro = min(wcro, ww)
        rr   = random.randint(0, hh-hcro)
        cc   = random.randint(0, ww-wcro)
        
        # Crop patch
        for modality in self.required:
            imgs[modality] = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in imgs[modality]]  
            imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]

        if self.noise:
            noise_level = self.noise * random.random()
            for modality in self.required:
                if modality == 'blur' or modality == 'turb':
                    imgs[modality] = [self._inject_noise(img, noise_level) for img in imgs[modality]]  
            
        aug    = random.randint(0, 8)
        # Data Augmentations
        for modality in self.required:
            if aug==1:
                imgs[modality] = [img.flip(1) for img in imgs[modality]]
            elif aug==2:
                imgs[modality] = [img.flip(2) for img in imgs[modality]]
            elif aug==3:
                imgs[modality] = [torch.rot90(img, dims=(1,2)) for img in imgs[modality]]
            elif aug==4:
                imgs[modality] = [torch.rot90(img,dims=(1,2), k=2) for img in imgs[modality]]
            elif aug==5:
                imgs[modality] = [torch.rot90(img,dims=(1,2), k=3) for img in imgs[modality]]
            elif aug==6:
                imgs[modality] = [torch.rot90(img.flip(1), dims=(1,2)) for img in imgs[modality]]
            elif aug==7:
                imgs[modality] = [torch.rot90(img.flip(2), dims=(1,2)) for img in imgs[modality]]
        return imgs
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        if self.train:
            loaded_imgs = self._fetch_chunk_train(index_)
        else:
            loaded_imgs = self._fetch_chunk_val(index_)
            
        if self.require_turb and (not self.require_tilt) and (not self.require_blur):
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
            
        if (not self.require_turb) and (not self.require_tilt) and self.require_blur:
            return torch.stack(loaded_imgs['blur'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
            
        if (not self.require_turb) and self.require_tilt and (not self.require_blur):
            return torch.stack(loaded_imgs['tilt'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
            
        if self.require_turb and (not self.require_tilt) and self.require_blur:
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['blur'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
            
        if self.require_turb and self.require_tilt and (not self.require_blur):
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['tilt'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
        if self.require_turb and self.require_tilt and self.require_blur:
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['tilt'], dim=0), \
                    torch.stack(loaded_imgs['blur'], dim=0), torch.stack(loaded_imgs['gt'], dim=0)
        else:
            raise
        

class DataLoaderTurbVideoTest(Dataset):
    def __init__(self, data_path, info_path, turb=True, tilt=False, blur=False, blur2=False,
                    level='all', num_frames=120, patch_unit=16):
        super(DataLoaderTurbVideoTest, self).__init__()
        self.num_frames = num_frames
        self.require_turb = turb
        self.require_tilt = tilt
        self.require_blur = (blur or blur2)
        
        if blur2:
            self.blur_dir = os.path.join(data_path, 'test_blur2')
        else:
            self.blur_dir = os.path.join(data_path, 'test_blur')
        self.turb_dir = os.path.join(data_path, 'test_turb')
        self.tilt_dir = os.path.join(data_path, 'test_tilt')
        self.gt_dir = os.path.join(data_path, 'test_gt')
        
        seqs_info = json.load(open(info_path, 'r'))
        
        if level == 'all':
            self.seqs_info = {}
            count = 0
            for info in seqs_info.values():
                if type(info)==dict:
                    self.seqs_info[count] = info
                    count += 1
        else:
            self.seqs_info = {}
            count = 0
            for info in seqs_info.values():
                if type(info)==dict and info['turb_level'] == level:
                    self.seqs_info[count] = info
                    count += 1
                    
        self.sizex = len(self.seqs_info)
        self.env = {}
        self.txn = {}
        self.env['gt'] = lmdb.open(self.gt_dir, map_size=1099511627776)
        self.txn['gt'] = self.env['gt'].begin()
        self.required = ['gt']
        if self.require_turb:
            self.env['turb'] = lmdb.open(self.turb_dir, map_size=1099511627776)
            self.txn['turb'] = self.env['turb'].begin()
            self.required.append('turb')
        if self.require_tilt:
            self.env['tilt'] = lmdb.open(self.tilt_dir, map_size=1099511627776)
            self.txn['tilt'] = self.env['tilt'].begin()
            self.required.append('tilt')
        if self.require_blur:
            self.env['blur'] = lmdb.open(self.blur_dir, map_size=1099511627776)
            self.txn['blur'] = self.env['blur'].begin()
            self.required.append('blur')
        
        self.pu = patch_unit

    def __len__(self):
        return self.sizex
        
    def _fetch_chunk_val(self, idx):
        ps = self.ps
        info = self.seqs_info[idx]
        vname = info['video_name']
        h, w, total_frames = info['h'], info['w'], info['length'] 
        c = 3
        if total_frames < self.num_frames:
            print('no enough frame in video ' + self.gt_list[idx])
        start_frame_id = random.randint(0, total_frames-self.num_frames)
        
        # load frames from database
        imgs = {}
        for modality in self.required:
            imgs[modality] = []
        for fi in range(start_frame_id, start_frame_id+self.num_frames):
            key = '{:s}_{:05d}'.format(vname, fi)
            for modality in self.required:
                enc_key = self.txn[modality].get(key.encode())
                load_img = np.frombuffer(enc_key, dtype='uint8')
                imgs[modality].append(load_img.reshape(h,w,c))
        
        for modality in self.required:
            imgs[modality] = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs[modality]]   
        
        if ps > 0:
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
            for modality in self.required:
                if padw!=0 or padh!=0:
                    imgs[modality] = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in imgs[modality]]   
                imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]

            hh, ww = imgs['gt'][0].shape[1], imgs['gt'][0].shape[2]
            rr, cc = (hh-ps) // 2, (ww-ps) // 2
            # Crop patch
            for modality in self.required:
                imgs[modality] = [img[:, rr:rr+ps, cc:cc+ps] for img in imgs[modality]]
        else:
            for modality in self.required:
                imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]
        
        if self.noise:
            noise_level = self.noise * random.random()
            for modality in self.required:
                if modality == 'blur' or modality == 'turb':
                    imgs[modality] = [self._inject_noise(img, noise_level) for img in imgs[modality]]  
        return imgs
                         
    def _fetch_chunk(self, idx):
        pu = self.pu
        
        info = self.seqs_info[idx]
        vname = info['video_name']
        h, w, total_frames = info['h'], info['w'], info['length'] 
        c = 3
        load_frames = min(total_frames, self.num_frames)

        # load frames from database
        imgs = {}
        for modality in self.required:
            imgs[modality] = []
        for fi in range(load_frames):
            key = '{:s}_{:05d}'.format(vname, fi)
            for modality in self.required:
                enc_key = self.txn[modality].get(key.encode())
                load_img = np.frombuffer(enc_key, dtype='uint8')
                imgs[modality].append(load_img.reshape(h,w,c))
        
        for modality in self.required:
            imgs[modality] = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs[modality]]   
        
        padh, padw = 0, 0
        if h % pu != 0:
            padh = pu - h % pu
        if w % pu != 0:
            padw = pu - w % pu
        if padh + padw > 0:
            # left, top, right and bottom
            for modality in self.required:
                imgs[modality] = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in imgs[modality]]
        for modality in self.required:
            imgs[modality] = [TF.to_tensor(img) for img in imgs[modality]]
        return imgs, h, w, load_frames, vname
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        loaded_imgs, h, w, num_frames, path = self._fetch_chunk(index_)
            
        if self.require_turb and (not self.require_tilt) and (not self.require_blur):
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path
            
        if (not self.require_turb) and (not self.require_tilt) and self.require_blur:
            return torch.stack(loaded_imgs['blur'], dim=0), torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path
            
        if (not self.require_turb) and self.require_tilt and (not self.require_blur):
            return torch.stack(loaded_imgs['tilt'], dim=0), torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path

        if self.require_turb and (not self.require_tilt) and self.require_blur:
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['blur'], dim=0),\
                 torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path
                        
        if self.require_turb and self.require_tilt and (not self.require_blur):
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['tilt'], dim=0), \
                torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path
            
        if self.require_turb and self.require_tilt and self.require_blur:
            return torch.stack(loaded_imgs['turb'], dim=0), torch.stack(loaded_imgs['tilt'], dim=0), \
                    torch.stack(loaded_imgs['blur'], dim=0), torch.stack(loaded_imgs['gt'], dim=0), h, w, num_frames, path
        else:
            raise