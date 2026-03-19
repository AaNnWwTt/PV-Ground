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

## Training

## Evaluation

## Main Results

## Acknowledgements
We are quite grateful for [BUTD-DETR](https://github.com/nickgkan/butd_detr), [EDA](https://github.com/yanmin-wu/EDA), [MCLN](https://github.com/qzp2018/MCLN) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).


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
