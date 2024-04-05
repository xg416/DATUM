<div align="center">

# 【CVPR'2024🔥】Spatio-Temporal Turbulence Mitigation: A Translational Perspective
</div>

## [🔥 Project Page](https://xg416.github.io/DATUM/) | [Paper](https://arxiv.org/abs/2401.04244)

## 🧩 Dataset and Pre-train Models
| Datasets | Pre-train Models | 
|:-----: |:-----: |
| [ATSyn_dynamic](https://app.box.com/s/b1mhcatitus3poo2tgcelsd4rmw9o35o) | [Pretrained dynamic scene model](https://drive.google.com/file/d/1IClAWZ-9kY5TggmuQGp11_dqOgCOCbjx/view?usp=sharing) |
| [ATSyn_static](https://app.box.com/s/xbx844bppqls2cqi74apac49iln0ztbz)  | [Pretrained static scene model](https://drive.google.com/file/d/13pJyzXo3ricYIy8WHMAWMxp4crEXUolg/view?usp=sharing) |

## 🔑 Setup and Prepare LMDB files
```
cd code
pip install -r requirements.txt
```
LMDB format is encouraged for our dynamic scene data. Before preparing your LMDB file from the downloaded ATSyn-dynamic dataset, please ensure you have 2TB of free space on your server. You will need to modify the path variables in the *make_lmdb.py* file. After that, just run
```
cd code
python make_lmdb.py
```
Alternatively, you can also use the .mp4 files for training, please refer to the code in *data/dataset_video_train.py* and training script in [TMT](https://github.com/xg416/TMT) for more information.

## 🛠️ Training 
For the training on dynamic scene data, run the following:
```
python train_DATUM_dynamic.py --train_path ${your_training_data_path} --train_info ${the associated train_info.json} --val_path ${your_testing_data_path} --val_info ${the associated test_info.json} -f ${loaded_model_path (if you want to resume a pre-trained checkpoint)} 
```
Other arguments for training are described in the *train_DATUM_dynamic.py* file, please refer to them for more flexible training. Use smaller *patch_size* and *num_frames* in the beginning phase of training can accelerate the entire process.

Later, you can start finetuning on the static scene images for the static scene model by running the following:
```
python train_DATUM_static.py --train_path ${your_training_data_path} --val_path ${your_testing_data_path} -f ${pretrained_dynamic_scene_model_path} 
```
We injected a certain level of Gaussian noise during training in both modalities for better generalization on real-world data.

## 🚀 Performance Evaluation
Dynamic scene model on ATSyn_dynamic dataset:
```
python test_DATUM_dynamic.py --data_path ${your_testing_data_path} --val_info ${the associated test_info.json} -result ${path_for_stored_output} -mp ${testing_model_path} 
```
Static scene model on ATSyn_static dataset:
```
python test_DATUM_static.py --val_path ${your_testing_data_path} -result ${path_for_stored_output} -f ${testing_model_path} 
```
Inference on Turbulence Text dataset, we generate the central 4 frames for the text recognition:
```
python inference_DATUM_text.py -f ${testing_static_scene_model_path} --n_frames 60 --resize 360
```
Please modify the path of the input and output images in the *inference_DATUM_text.py* 

Please refer to *CWSSIM_static.py* and *CWSSIM_dynamic.py* to evaluate the CW-SSIM score.


## 👍 Useful Links
### [Turbulence @ Purdue i2Lab](https://engineering.purdue.edu/ChanGroup/project_turbulence.html) | [UG2 workshop @ CVPR 2024](https://cvpr2024ug2challenge.github.io/) 

### Restoration:
[Link](https://ieeexplore.ieee.org/abstract/document/10400926) Zhang, Xingguang, Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan. "Imaging through the atmosphere using turbulence mitigation transformer." IEEE Transactions on Computational Imaging (2024).

[Link](https://openaccess.thecvf.com/content/ICCV2023/html/Jaiswal_Physics-Driven_Turbulence_Image_Restoration_with_Stochastic_Refinement_ICCV_2023_paper.html) Ajay Jaiswal*, Xingguang Zhang*, Stanley H. Chan, Zhangyang Wang. "Physics-Driven Turbulence Image Restoration with Stochastic Refinement." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 12170-12181 

[Link](https://arxiv.org/abs/2206.02146) Liang, Jingyun, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V. Gool. "Recurrent video restoration transformer with guided deformable attention." Advances in Neural Information Processing Systems 35 (2022): 378-393.

[Link](https://www.nature.com/articles/s42256-021-00392-1) Jin, Darui, Ying Chen, Yi Lu, Junzhang Chen, Peng Wang, Zichao Liu, Sheng Guo, and Xiangzhi Bai. "Neutralizing the impact of atmospheric turbulence on complex scene imaging via deep learning." Nature Machine Intelligence 3, no. 10 (2021): 876-884.

### Datasets:
[OTIS dataset](https://zenodo.org/records/161439) | [TSRWGAN data](https://zenodo.org/records/5101910) | [Turbulence Text](https://drive.google.com/file/d/1QWvQfPM-lJwGqK_Wm6lDbi-tYBu-Uopq/view?usp=sharing) | [Heat Chamber](https://drive.google.com/file/d/14iVachB95bCCtke8ONPD9CCH20JO75v2/view?usp=sharing) | [TMT dataset](https://github.com/xg416/TMT) | [BRIAR](https://arxiv.org/abs/2211.01917) (Not public yet)


## 📘 Citation
Please consider citing our work as follows if it is helpful.
```
@InProceedings{zhang2024spatio,
    author={Zhang, Xingguang and Chimitt, Nicholas and Chi, Yiheng and Mao, Zhiyuan and Chan, Stanley H}, 
    title={Spatio-Temporal Turbulence Mitigation: A Translational Perspective},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2024}
}
```
