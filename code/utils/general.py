import os 
import torch
import datetime
from collections import OrderedDict

def get_time(path):
    tstring = path.split('_')[-1]
    return datetime.datetime.strptime(tstring,'%m-%d-%Y-%H-%M-%S')
    
def find_latest_checkpoint(all_log_dir, run_name):
    run_name_list = [v for v in os.listdir(all_log_dir) if v.startswith(run_name)]
    run_name_list.sort(reverse=True, key=get_time)
    for p in run_name_list:
        ckpt_path = os.path.join(all_log_dir, p, 'checkpoints', 'latest.pth')
        if os.path.exists(ckpt_path):
            return ckpt_path
    return None
    
    
def create_log_folder(dir_name):
    os.mkdir(dir_name)
    path_imgs = os.path.join(dir_name, 'imgs')
    os.mkdir(path_imgs)
    path_ckpt = os.path.join(dir_name, 'checkpoints')
    os.mkdir(path_ckpt)
    path_scipts = os.path.join(dir_name, 'scipts')
    os.mkdir(path_scipts)
    return path_imgs, path_ckpt, path_scipts


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return checkpoint, model
    

def get_cuda_info(logger):
    cudnn_version = torch.backends.cudnn.version()
    count = torch.cuda.device_count()
    device_name_0 = torch.cuda.get_device_name(0)
    memory_0 = torch.cuda.get_device_properties(0).total_memory/(1024**3)

    logger.info(f'__CUDNN VERSION: {cudnn_version}\n'
        f'__Number CUDA Devices: {count}\n'
        f'__CUDA Device Name: {device_name_0}\n'
        f'__CUDA Device Total Memory [GB]: {memory_0}')
    
