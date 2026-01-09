# Codes for "HySpeFAS: A Hyperspectral Face Anti-spoofing Dataset based on Snapshot Compressive Imaging"

## Requirements:
```requirements.txt
opencv-python~=4.6.0.66
scipy~=1.8.1
numpy~=1.23.0
torch~=1.12.0
torchvision~=0.13.0
tqdm~=4.64.0
torcheval~=0.0.7
torchmetrics~=1.4.0
```

The codes have been tested under Python 3.8 on Ubuntu 20.04 with RTX2080Ti GPU and CUDA 11.2.

## Overview
This repository provides the baseline methods and models proposed in our paper.
- The HySpeFAS dataset should be downloaded and put in the _data_ folder. 
- The _fas_models_ folder provides all the tested baseline models in the paper.
- The _reconstruction_ folder contains the codes for reconstructing HSIs from SSIs and the sensing matrix using convex optimization.

## Dataset Downloading
- The dataset is under CC BY-NC-ND 4.0 license.
- The dataset can be accessed thourgh [Google Drive](https://drive.google.com/drive/folders/1ykd0iKoYSfy1Vf72h09guljsWqeWnCrs?usp=sharing)
or [Baidu Drive](https://pan.baidu.com/s/1650Q9GBrMc-nNOsak7H--Q?pwd=68hx)
- Please sign and send the agreement according to the instructions and then get the unzip password.
- Download the full dataset and unzip the files.

## Usage

### Preparation
- Install the requirements.
- Download the HySpeFAS dataset from the download link.
- Unzip the dataset files and put them in the _data_ directory.
- Check if the dataset path in the code matches the actual data directory.
- (Optional) Run the script _reconstruction/reconstruct_images.m_ to get the reconstructed HSIs (R-HSIs). This is essential 
to train models based on R-HSIs such as TwIST-ResNet.

### Baselines for Track 1:
For training SMN-ResNet, run the following command:
```shell
python train_ssi.py
```
For training TwIST-ResNet, run the following command:
```shell
python train_hsi.py
```
Other SSI-based baseline models can be also found in _fas_models/ssi_models_ and trained using train_ssi.py

### Baselines for Track 2:
For training SSI-RGB fusion network, run the following command:
```shell
python train_ssirgb_fusion.py
```
For training HSI-RGB fusion network, run the following command:
```shell
python train_hsirgb_fusion.py
```

### Baselines for Track 3:
Run the following command to train the baseline model:
```shell
python train_rec.py
```

## Citation
```text
@ARTICLE{11314673,
  author={Rao, Shijie and Huang, Yidong and Zhang, Xueqian and Fang, Hao and Liu, Ajian and Wan, Jun and Cui, Kaiyu and Li, Yali},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={HySpeFAS: A Hyperspectral Face Anti-Spoofing Dataset Based on Snapshot Compressive Imaging}, 
  year={2026},
  volume={21},
  number={},
  pages={669-680},
  keywords={Hyperspectral imaging;Face recognition;Imaging;Three-dimensional printing;Metasurfaces;Faces;System-on-chip;Hardware;Cameras;Image coding;Snapshot hyperspectral imaging;face anti-spoofing;compressive imaging;deep neural network},
  doi={10.1109/TIFS.2025.3648158}}

```
