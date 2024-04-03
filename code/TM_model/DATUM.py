import sys
import os
# sys.path.insert(0, '/home/zhan3275/.local/lib/python3.8/site-packages')

# my_env = os.environ.copy()
# my_env["PATH"] = "/home/zhan3275/.local/bin:" + my_env["PATH"]
# os.environ.update(my_env)

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from einops import rearrange
import argparse
from .archs import conv1x1, conv3x3, conv5x5, actFunc, flow_warp, SpyNet
from .op.deform_attn import deform_attn, DeformAttnPack



class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, self.in_channels, kernel_size=(1, 1),
                      padding=(0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            RDB(self.in_channels, growthRate=16, num_layer=3, activation='gelu'),
            nn.Conv2d(self.in_channels, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=1, padding=0),)
        
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = self.in_channels * 2
        self.proj_q = conv1x1(self.in_channels, self.proj_channels)
        self.proj_k = conv1x1(self.in_channels, self.proj_channels)
        self.proj_v = conv1x1(self.in_channels, self.proj_channels)
        self.proj = conv1x1(self.proj_channels, self.in_channels)
        self.ffn = FeedForward(self.in_channels, 2)

    def init_offset(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()
    #new_f = self.deform_att(f3[:,i,...], f3[:,i-1,...], s, align_f, flows_backward[:,i-1,...])
    def forward(self, f, f_aligned, r, flows):
        offset = self.conv_offset(torch.cat([f_aligned, r, flows], dim=1))
        offset = offset + flows.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        
        b, c, h, w = offset.shape
        
        q = self.proj_q(r).unsqueeze(1)
        k = self.proj_k(f).unsqueeze(1)
        v = self.proj_v(f).unsqueeze(1)
        kv = torch.cat([k, v], 2)
        # print(q.shape, kv.shape, offset.shape)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size)
        # print(v.shape)
        v = self.proj(v.squeeze(1))
        v = v + self.ffn(v)

        return v, offset.view(b, c // 2, 2, h, w).mean(1).flip(1)
        

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=[3,3], reduction=16, activation='gelu'):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[0], padding=kernel_size[0]//2, bias=True))
        modules_body.append(actFunc(activation))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[1], padding=kernel_size[1]//2, bias=True))

        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
        
# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# RDB fusion module
class RDB_shrink(nn.Module):
    def __init__(self, in_channels, med_channels, growthRate, num_layer, activation='relu'):
        super(RDB_shrink, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.shrink = conv3x3(in_channels, med_channels, stride=1)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.shrink(x)
        return out
        
        
# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, med_channels, out_channel, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        self.RDBs.append(RDB_shrink(in_channels, med_channels, growthRate, num_layer, activation))
        for i in range(num_blocks-1):
            self.RDBs.append(RDB(med_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * med_channels, med_channels)
        self.conv3x3 = conv3x3(med_channels, out_channel)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out

        
# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, addition_channels=0, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels+addition_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels+addition_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out

class AttentionCT(nn.Module):
    def __init__(self, dim, num_heads, out_dim):
        super(AttentionCT, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.get_qkv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                            nn.Conv2d(dim, dim*3, kernel_size=1, bias=False))
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=False)


    def forward(self, x):
        b,ct,h,w = x.shape
        
        qkv = self.get_qkv(x)
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head ct) h w -> b head ct (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head ct) h w -> b head ct (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head ct) h w -> b head ct (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head ct (h w) -> b (head ct) h w', h=h, w=w)
        out = self.project_out(out)
        return out
        
# Global temporal attention module
class TCA(nn.Module):
    def __init__(self, para):
        super(TCA, self).__init__()
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.act = actFunc(para.activation)
        self.pre_fusion = conv1x1(2*5*self.related_f*self.n_feats, 5*self.related_f*self.n_feats)
        self.att = AttentionCT(5*self.related_f*self.n_feats, num_heads=8, out_dim=5*self.related_f*self.n_feats)
        
    def forward(self, h):
        # h: [(n=4,c=800,h=60,w=60), ..., (n,c,h,w)]
        # put the interest frame to the tail to mark its position
        x = self.act(self.pre_fusion(h))
        out = x + self.att(x)
        return out
        

class ConvGRUCell(nn.Module):
    # based on https://github.com/bionick87/ConvGRUCell-pytorch
    def __init__(self, input_size, hidden_size, kernel_size, act):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, \
                                     self.kernel_size, padding=self.kernel_size//2)
        self.Conv_ct = nn.Sequential(
            conv3x3(self.input_size + self.hidden_size, self.hidden_size),
            RDB(in_channels=self.hidden_size, growthRate=self.hidden_size, num_layer=3, activation=act),
            conv3x3(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, input, hidden):
        c1           = self.ConvGates(torch.cat((input, hidden),1))
        (reset_gate, update_gate) = torch.sigmoid(c1).chunk(2, 1)
        gated_hidden = reset_gate * hidden
        p1           = self.Conv_ct(torch.cat((input, gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = update_gate * hidden + (1-update_gate) * ct
        return next_h

# RDB-based GRU cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = 15
        self.F_R = RDNet(in_channels=(5 + 5 + 5)*self.n_feats, med_channels=6*self.n_feats, out_channel=5*self.n_feats, 
                            growthRate=self.n_feats, num_layer=3, num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        # F_h: hidden state part
        self.F_h = ConvGRUCell(5*self.n_feats, 5*self.n_feats, 3, self.activation)

    def forward(self, x, x_, s_last):
        '''
        0: torch.Size([2, 32, 256, 256])
        1: torch.Size([2, 64, 128, 128])
        2: torch.Size([2, 96, 64, 64])
        3: torch.Size([2, 80, 64, 64])
        '''
        out = self.F_R(torch.cat([x, x_, s_last], dim=1))
        s = self.F_h(out, s_last)
        
        return out, s
        

class Encoder(nn.Module):
    def __init__(self, para):
        super(Encoder, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.F_B01 = conv5x5(3, self.n_feats, stride=1)
        self.F_B02 = conv5x5(3, self.n_feats, stride=2)
        self.F_B03 = conv5x5(3, self.n_feats, stride=2)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           addition_channels=self.n_feats, activation=self.activation)

    def forward(self, x):
        '''
        0: torch.Size([2, 32, 256, 256])
        1: torch.Size([2, 64, 128, 128])
        2: torch.Size([2, 96, 64, 64])
        '''
        b,l,_,h,w = x.shape
        x = x.flatten(0,1)
        
        out1 = self.F_B01(x)
        inp2 = self.F_B02(x)
        inp3 = self.F_B03(F.interpolate(x, size=(h//2, w//2), mode='bilinear', align_corners=False))
        out2 = self.F_B1(out1)
        out3 = self.F_B2(torch.cat([out2, inp2], 1))
        out3 = torch.cat([out3, inp3], dim=1)
        
        return out1.unflatten(0, (b, l)), out2.unflatten(0, (b, l)), out3.unflatten(0, (b, l))



# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para, out_dim=3):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.up3 = nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 4 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        self.fusion2 = CAB(6*self.n_feats, [1,3])
        
        self.up2 = nn.ConvTranspose2d(6*self.n_feats, 2*self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)                
        self.fusion1 = CAB(3*self.n_feats, [1,3])               
        self.output = nn.Sequential(
            conv3x3(3*self.n_feats, self.n_feats, stride=1),
            conv3x3(self.n_feats, out_dim, stride=1)
        )

    def forward(self, x3, x2, x1):
        # channel: 10, 2, 1 * n_feat
        x3_up = self.up3(x3)
        x2 = self.fusion2(torch.cat([x3_up, x2], 1))
        
        x2_up = self.up2(x2)
        x1 = self.fusion1(torch.cat([x2_up, x1], 1))
        return self.output(x1)


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.wl = para.future_frames + para.past_frames + 1
        self.ds_ratio = 4
        self.device = torch.device('cuda')

        self.spynet = SpyNet(para.spynet_path)
        self.get_features = Encoder(para)
        self.propogate = RDBCell(para)
        self.fusion = TCA(para)
        self.recons = Reconstructor(para)
        self.reconsT = Reconstructor(para, 2)
        self.deform_att = GuidedDeformAttnPack(5*self.n_feats, 5*self.n_feats, attention_window=[3, 3], \
                                                attention_heads=8, deformable_groups=16, clip_size=1, max_residue_magnitude=10)
        self.output_full = para.output_full
        
        
    def forward(self, inp):
        x, xT = inp
        outputs, outputs_removeT = [], []
        hf, hl = [], []
        batch_size, frames, channels, height, width = x.shape

        
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        flows_backward, flows_forward = self.get_flow(x, resize=(s_height, s_width))
        f1, f2, f3 = self.get_features(x)
        
        # for i in range(frames-1):
        #     print(i, x[:,i,...].sum().item(), flows_backward[:,i,...].sum().item(), flows_forward[:,i,...].sum().item())
        #     np.save(f"/home/zhan3275/data/temp/img_{i}.npy", x[:,i,...].cpu().numpy())
        #     np.save(f"/home/zhan3275/data/temp/flow_f_{i}.npy", flows_forward[:,i,...].cpu().numpy())
        #     np.save(f"/home/zhan3275/data/temp/flow_b_{i}.npy", flows_backward[:,i,...].cpu().numpy())
            
        # forward h structure: (batch_size, channel, height, width)
        r = torch.zeros(batch_size, 5*self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            if i > 0:
                flow_update = flow_r + flow_warp(flows_backward[:,i-1,...], flow_r)
                # print('f', i, flow_r.sum().item(), flows_backward[:,i-1,...].sum().item(), flow_update.sum().item())
                align_f = flow_warp(f3[:,i,...], flow_update, 'bilinear') # align frame i to i-1
                new_f, flow_r = self.deform_att(f3[:,i,...], align_f, r, flow_update)
            else:
                new_f = f3[:,i,...]
                flow_r = torch.zeros_like(flows_backward[:,0,...])
            h, r = self.propogate(new_f, f3[:,i,...], r)
            hf.append(h)
            
        r = torch.zeros(batch_size, 5*self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames-1, -1, -1):
            if i < frames-1:
                flow_update = flow_r + flow_warp(flows_forward[:,i,...], flow_r)
                # print('b', i, flow_r.sum().item(), flows_forward[:,i,...].sum().item(), flow_update.sum().item())
                align_f = flow_warp(f3[:,i,...], flow_update, 'bilinear') # align frame i to i+1
                new_f, flow_r = self.deform_att(f3[:,i,...], align_f, r, flow_update)
            else:
                new_f = f3[:,i,...]
                flow_r = torch.zeros_like(flows_forward[:,0,...])
            h, r = self.propogate(new_f, f3[:,i,...], r)
            hl.insert(0, torch.cat([hf[i], h], 1))

        # print(h_all.shape)
        h_all = self.construct_h(torch.stack(hl, dim=1), frames)
        fused = self.fusion(h_all) # b*f, c, h, w
        if self.output_full:
            f2 = torch.cat(list(f2.transpose(0,1)), 0)
            f1 = torch.cat(list(f1.transpose(0,1)), 0)
            xT = torch.cat(list(xT.transpose(0,1)), 0)
        else:
            f2 = torch.cat(list(f2[:, self.num_fb:frames - self.num_ff, ...].transpose(0,1)), 0)
            f1 = torch.cat(list(f1[:, self.num_fb:frames - self.num_ff, ...].transpose(0,1)), 0)
            xT = torch.cat(list(xT[:, self.num_fb:frames - self.num_ff, ...].transpose(0,1)), 0)   
        # TODO: change the list and cat to view function
        ReverseT = self.reconsT(fused, f2, f1)
        ReverseT2 = F.interpolate(ReverseT/2, size=(height//2, width//2), mode='bilinear', align_corners=False)
        out_removeT = flow_warp(xT, ReverseT)
        out = self.recons(fused, flow_warp(f2, ReverseT2), flow_warp(f1, ReverseT))
        out_removeT = torch.stack(out_removeT.chunk(out_removeT.shape[0]//batch_size, 0), 1)
        out = torch.stack(out.chunk(out.shape[0]//batch_size, 0), 1)
        return out, out_removeT

    def get_flow(self, x, resize):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        n, t, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        if (h, w) != resize:
            x_1 = F.interpolate(x_1, size=resize, mode='bilinear', align_corners=False)
            x_2 = F.interpolate(x_2, size=resize, mode='bilinear', align_corners=False)
            h, w = resize
            
        flows_backward = self.spynet(x_1, x_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def construct_h(self, h, frames):
        # h: b, f, c, h, w
        nhl = []
        if self.output_full:
            for i in range(frames):
                if i < self.num_fb:
                    fl = [n for n in range(self.wl)]
                    fl.append(fl.pop(i))
                elif i >= frames - self.num_ff:
                    fl = [n for n in range(frames-self.wl, frames)]
                    fl.append(fl.pop(i-frames))
                else:
                    fl = [n for n in range(i - self.num_fb, i + self.num_ff + 1)]
                    fl.append(fl.pop(self.wl//2))
                nhl.append(h[:, fl, ...].flatten(1,2))
        else:
            for i in range(self.num_fb, frames - self.num_ff):
                fl = [n for n in range(i - self.num_fb, i + self.num_ff + 1)]
                fl.append(fl.pop(self.wl//2))
                nhl.append(h[:, fl, ...].flatten(1,2))
        return torch.cat(nhl, 0)

def feed(model, iter_samples):
    inputs = iter_samples
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--model', type=str, default='TMRNN', help='type of model to construct')
    parser.add_argument('--spynet_path', type=str, default="/home/zhan3275/turb/recon/RNN/DATUM/model_zoo/spynet_turb.pth")
    parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_full', action='store_false', help='# input frames = # output frames')
    return parser.parse_args()
    
    
if __name__ == "__main__":
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import time
    from thop import profile, clever_format
 
    params = get_args()
    model = Model(params).cuda()
    device = torch.device('cuda')
    data = torch.randn((1,8,3,540,960)).to(device=device)
    data2 = torch.randn((1,8,3,540,960)).to(device=device)
    # for i in range(20):
    #     out = model(data)
    s = time.time()
    with torch.no_grad():
        out = model((data, data2))
        for i in range(10):
            out = model((data, data2))
    print(time.time()-s)
    
    # with torch.no_grad():
    #     out = model(data)
    # print(out.shape)
    
    # with torch.no_grad():
    #     summary(model, (12,3,256,256))
    
    # with torch.no_grad():
    #     s = time.time()
    #     for i in range(1):
    #         print('{:>16s} : {:<.4f} [M]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 6))
    #         flops = FlopCountAnalysis(model, data)
    #         print(flop_count_table(flops))
    #         print(flops.total())
    #         # net(inputs)
    #     print(time.time()-s)
    
    flops, params = profile(model, inputs=((data, data2),), verbose=False)
    
    macs, params = clever_format([flops/8, params], "%.3f")
    print(macs, params)
    
