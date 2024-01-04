import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simulator(nn.Module):
    def __init__(self, path, H, W):
        super().__init__()
        
        # Loading the P2S module, integral_path, blur kernels
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.mapping = _P2S()
        self.mapping.load_state_dict(
            torch.load(os.path.join(path, 'dfp2s_model/P2S_state3.pt'), map_location=self.device))
        self.mapping = self.mapping.to(device=self.device)
        dict_psf = torch.load(os.path.join(path, 'dfp2s_model/dictionary3.pt'), map_location=self.device)
        # dict_psf = torch.load(os.path.join(path, '/home/xingguang/Documents/turb/TurbulenceSim_P2S/P2Sv3/data/dictionary.pt'), map_location=self.device)
        self.mu = dict_psf['mu'].unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.basis_psf = dict_psf['dictionary'].unsqueeze(1).to(self.device, dtype=torch.float32)
        self.mu_norm = torch.sum(self.mu.abs(), dim=(-2,-1), keepdim=True)
        self.basis_norm = torch.sum(self.basis_psf.abs(), dim=(-2,-1), keepdim=True)    
        yy, xx = torch.meshgrid(torch.arange(0, H, device=self.device), torch.arange(0, W, device=self.device))
        self.grid = torch.stack((xx, yy), -1).unsqueeze(0).to(dtype=torch.float)
        self.p2s_blur = self._blur_construction(67)

    def _blur_construction(self, ksize):
        local_mu = F.interpolate(self.mu, size=(ksize, ksize), mode='bilinear')
        local_basis_psf = F.interpolate(self.basis_psf, size=(ksize, ksize), mode='bilinear')
        local_mu *= self.mu_norm / torch.sum(local_mu.abs(), dim=(-2,-1), keepdim=True)
        local_basis_psf *= self.basis_norm / torch.sum(local_basis_psf.abs(), dim=(-2,-1), keepdim=True)
        p2s_blur = nn.Conv2d(101, 101, ksize, groups=101, padding='same', padding_mode='reflect', bias=False, device=self.device, dtype=torch.float32)
        p2s_blur.weight.data[:100, ...] = local_basis_psf
        p2s_blur.weight.data[100, ...] = local_mu
        return p2s_blur

    def forward(self, img, zernike, ksize=67):
        p2s_blur = self.p2s_blur
        # adapt to new input size
        batchN, channelN, H, W = img.shape
        # zernike = zernike.to(torch.float32)
        self.grid = self.grid.expand(batchN, -1, -1, -1)
        
        # Generating the pixel-shift values
        pos = zernike[...,:2]
        flow = 2.0 * (self.grid + pos*1)/(torch.tensor((W, H), device=self.device)-1) - 1.0
        # applying the flow array
        tilt_img = F.grid_sample(img, flow, 'bilinear', padding_mode='border', align_corners=False)
        tilt_img = tilt_img.view((-1, 1, H, W))

        weight = torch.ones((batchN, H, W, 101), dtype=torch.float32, device=self.device)
        weight[...,:100] = self.mapping(zernike[..., 2:])
        ones_img = torch.ones_like(tilt_img)
        big_img = torch.cat((tilt_img.view(batchN, channelN, H, W).unsqueeze(4), 
                                ones_img.view(batchN, channelN, H, W).unsqueeze(4)), 1)
        big_img = big_img * weight.unsqueeze(1)
        dict_img = p2s_blur(big_img.view(-1, H, W, 101).permute(0,3,1,2)).view(batchN, -1, 101, H, W)
        norm_img = dict_img[:, 3:]
        out = torch.sum(dict_img[:, :3], dim=2) / torch.sum(norm_img, dim=2)
        return out


        
class _P2S(nn.Module):
    def __init__(self, input_dim=33, output_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        out = self.out(y)
        return out


if __name__ == "__main__":
    import cv2
    import numpy as np

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    zernike_path = '/home/xgz/Documents/turb/datasets/FFHQ256/Zernike/00010/0.pt'
    img_path = '/home/xgz/Documents/turb/datasets/FFHQ256/gt/00010.png'
    sim_path = '/home/xgz/Documents/turb/simulator/Nickv4/'
    x = cv2.imread(img_path) / 255
    im_input = torch.tensor(x.transpose((2,0,1)), device = device, dtype=torch.float32).unsqueeze(0)
    simulator = Simulator(sim_path, 256, 256).to(device, dtype=torch.float32)
    zer = torch.load(zernike_path, map_location=device)
    print(zer.shape)
    im_input = im_input.expand(4,-1,-1,-1)
    y = simulator(im_input, zer, 55)
    print(y.shape)
    turb = y[1].squeeze().permute(1,2,0).detach().cpu().numpy()
    turb = np.clip(turb, 0, 1)
    cv2.imwrite("try.png", turb*255)