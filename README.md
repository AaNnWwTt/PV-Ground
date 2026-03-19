# PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding
This repo is the official implementation of "**PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding**". [CVPR2026]

We are currently preparing the source code. It will be made available before the conference.

## Installation

+ **(1)** Create environment:
```
conda create -n pvg python=3.12
conda activate pvg
pip3 install torch torchvision
pip install numpy ipython psutil traitlets transformers<5.0 termcolor ipdb scipy tensorboardX h5py wandb plyfile tabulate einops six
```

+ **(2)** Install spacy for text parsing
```
pip install spacy
python -m spacy download en_core_web_sm
cd pointnet2
pip install . --no-build-isolation
```
+ **(3)** Install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): 
```
pip install spconv-cu124
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -e . --no-build-isolation
```
Note: You may need to comment out all the datasets in OpenPCDet/pcdet/datasets/__init__.py that cause errors during training and evaluation.

## Data Preparation

The final required files are as follows:
```
├── [DATA_ROOT]
│	├── [1] train_v3scans.pkl # Packaged ScanNet training set
│	├── [2] val_v3scans.pkl   # Packaged ScanNet validation set
│	├── [3] ScanRefer/        # ScanRefer utterance data
│	│	│	├── ScanRefer_filtered_train.json
│	│	│	├── ScanRefer_filtered_val.json
│	│	│	└── ...
│	├── [4] ReferIt3D/        # NR3D/SR3D utterance data
│	│	│	├── nr3d.csv
│	│	│	├── sr3d.csv
│	│	│	└── ...
│	├── [5] group_free_pred_bboxes/  # detected boxes (optional)
│	├── [6] gf_detector_l6o256.pth   # pointnet++ checkpoint (optional)
│	├── [7] roberta-base/     # roberta pretrained language model
│	├── [8] checkpoints/      # mcln pretrained models
```

+ **[1] [2] Prepare ScanNet Point Clouds Data**
  + **1)** Download ScanNet v2 data. Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission, and you will get the official download script `download-scannet.py`. Then use the following command to download the necessary files:
    ```
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.labels.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type .aggregation.json
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.0.010000.segs.json
    python2 download-scannet.py -o [SCANNET_PATH] --type .txt
    ```
    where `[SCANNET_PATH]` is the output folder. The scannet dataset structure should look like below:
    ```
    ├── [SCANNET_PATH]
    │   ├── scans
    │   │   ├── scene0000_00
    │   │   │   ├── scene0000_00.txt
    │   │   │   ├── scene0000_00.aggregation.json
    │   │   │   ├── scene0000_00_vh_clean_2.ply
    │   │   │   ├── scene0000_00_vh_clean_2.labels.ply
    │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene.......
    ```
  + **2)** Package the above files into two .pkl files(`train_v3scans.pkl` and `val_v3scans.pkl`):
    ```
    python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
    ```
+ **[3] ScanRefer**: Download ScanRefer annotations following the instructions [HERE](https://github.com/daveredrum/ScanRefer). Unzip inside `[DATA_ROOT]`.
+ **[4] ReferIt3D**: Download ReferIt3D annotations following the instructions [HERE](https://github.com/referit3d/referit3d). Unzip inside `[DATA_ROOT]`.
+ **[5] group_free_pred_bboxes**: Download [object detector's outputs](https://drive.google.com/drive/folders/1vfOeTLKdW2AFoQPoivxT5sFloeZSXnEf). Unzip inside `[DATA_ROOT]`. (not used in single-stage method)
+ **[6] gf_detector_l6o256.pth**: Download PointNet++ [checkpoint](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8) into `[DATA_ROOT]`.
+ **[7] roberta-base**: Download the roberta pytorch model:
  ```
  cd [DATA_ROOT]
  git clone https://huggingface.co/roberta-base
  cd roberta-base
  rm -rf pytorch_model.bin
  wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
  ```
+ **[8] checkpoints**: Our pre-trained models (see 3. Models).
+ **[9] ScanNetv2**: Prepare the preporcessed ScanNetv2 dataset follow "Data Preparation" section from https://github.com/sunjiahao1999/SPFormer, obtaining the dataset file with the following structure:
```
ScanNetv2
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
+ **[10] superpoints**: Prepare superpoints for each scene preprocessed from Step. 9.
  ```
  cd [DATA_ROOT]
  python superpoint_maker.py  # modify data_root & split
  ```
  
## Training

## Evaluation

## Main Results

## Acknowledgements
This repository is built on reusing codes of [BUTD-DETR](https://github.com/nickgkan/butd_detr), [EDA](https://github.com/yanmin-wu/EDA), [MCLN](https://github.com/qzp2018/MCLN) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). We are quite grateful for these works. 


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{shang2026pv,
  title={PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding},
  author={Shang, Junpeng and Shao, Feifei and Xiao, Jun and Li, Lin and Wang, Hongwei and Ma, Dongfang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
