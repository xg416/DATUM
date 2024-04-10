import os
import cv2
import lmdb
import json


dataset_name = 'test'
data_root = '/home/zhan3275/data/ATSyn_dynamic' # replace with your own path
save_root = '/home/zhan3275/data/ATSyn_LMDB'  # replace with your own path
data_path = os.path.join(data_root, dataset_name)
lmdb_path = os.path.join(save_root, dataset_name+'_lmdb')
os.makedirs(lmdb_path, exist_ok=True)


video_dir = f'/home/zhan3275/data/DATUM_dynamic/{dataset_name}/gt'  # replace with your own path
param_dir = f'/home/zhan3275/data/DATUM_dynamic/{dataset_name}/turb_param/'  # replace with your own path
seqs_info = {}
video_count = 0
length = 0
for vname in os.listdir(video_dir):
    if not vname.endswith('.mp4'):
        continue
    seq_info = {}
    param_path = os.path.join(param_dir, vname.replace('mp4', 'json'))
    turb_param = json.load(open(param_path, 'r'))
    seq_info['video_name'] = vname
    seq_info['turb_level'] = turb_param['level']
    seq_info['blur_kernel'] = turb_param['kernel_size']
    seq_info['temp_corr'] = turb_param['temp_corr']
    video_path = os.path.join(video_dir, vname)
    video = cv2.VideoCapture(video_path)
    h, w = int(video.get(4)), int(video.get(3))
    length_temp = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    seq_info['length'] = length_temp
    seq_info['h'] = h
    seq_info['w'] = w
    length += length_temp
    seqs_info[video_count] = seq_info
    video.release()
    video_count += 1
seqs_info['num_videos'] = video_count
seqs_info['num_frames'] = length
f = open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'w')
json.dump(seqs_info, f)
f.close()


# seqs_info = json.load(open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'r'))

for modality in ['gt', 'tilt', 'turb']:
    video_dir = os.path.join(data_root, dataset_name, modality)
    dn = dataset_name + '_' + modality
    for i in range(seqs_info['num_videos']):
        env = lmdb.open(os.path.join(lmdb_path, '{}'.format(dn)), map_size=1099511627776)
        txn = env.begin(write=True)
        vname = seqs_info[str(i)]['video_name']
        print(i, vname)
        video_path = os.path.join(video_dir, vname)
        video = cv2.VideoCapture(video_path)
        flag, frame = video.read()
        frame_idx = 0
        while flag:
            key = '%s_%05d' % (vname, frame_idx)
            txn.put(key=key.encode(), value=frame)
            flag, frame = video.read()
            frame_idx += 1
        video.release()
        txn.commit()
        env.close()


dataset_name = 'train'
data_root = '/home/zhan3275/data/ATSyn_dynamic/' # replace with your own path
save_root = '/home/zhan3275/data/ATSyn_LMDB'  # replace with your own path
data_path = os.path.join(data_root, dataset_name)
lmdb_path = os.path.join(save_root, dataset_name+'_lmdb')
os.makedirs(lmdb_path, exist_ok=True)

video_dir = f'/home/zhan3275/data/DATUM_dynamic/{dataset_name}/gt'  # replace with your own path
param_dir = f'/home/zhan3275/data/DATUM_dynamic/{dataset_name}/turb_param/'  # replace with your own path
seqs_info = {}
video_count = 0
length = 0
for vname in os.listdir(video_dir):
    if not vname.endswith('.mp4'):
        continue
    seq_info = {}
    param_path = os.path.join(param_dir, vname.replace('mp4', 'json'))
    turb_param = json.load(open(param_path, 'r'))
    seq_info['video_name'] = vname
    seq_info['turb_level'] = turb_param['level']
    seq_info['blur_kernel'] = turb_param['kernel_size']
    seq_info['temp_corr'] = turb_param['temp_corr']
    video_path = os.path.join(video_dir, vname)
    video = cv2.VideoCapture(video_path)
    h, w = int(video.get(4)), int(video.get(3))
    length_temp = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    seq_info['length'] = length_temp
    seq_info['h'] = h
    seq_info['w'] = w
    length += length_temp
    seqs_info[video_count] = seq_info
    video.release()
    video_count += 1
seqs_info['num_videos'] = video_count
seqs_info['num_frames'] = length
f = open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'w')
json.dump(seqs_info, f)
f.close()

# seqs_info = json.load(open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'r'))

for modality in ['gt', 'tilt', 'turb']:
    video_dir = os.path.join(data_root, dataset_name, modality)
    dn = dataset_name + '_' + modality
    for i in range(seqs_info['num_videos']):
        env = lmdb.open(os.path.join(lmdb_path, '{}'.format(dn)), map_size=1099511627776)
        txn = env.begin(write=True)
        vname = seqs_info[str(i)]['video_name']
        print(i, vname)
        video_path = os.path.join(video_dir, vname)
        video = cv2.VideoCapture(video_path)
        flag, frame = video.read()
        frame_idx = 0
        while flag:
            key = '%s_%05d' % (vname, frame_idx)
            txn.put(key=key.encode(), value=frame)
            flag, frame = video.read()
            frame_idx += 1
        video.release()
        txn.commit()
        env.close()
