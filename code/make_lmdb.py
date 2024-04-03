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

seqs_info = json.load(open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'r'))

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

seqs_info = json.load(open(os.path.join(lmdb_path,'{}_info.json'.format(dataset_name)), 'r'))

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


