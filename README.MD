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
- The _fas_models_ folder provides all of the tested baseline models in the paper.
- The _reconstruction_ folder contains the codes for reconstructing HSIs from SSIs and the sensing matrix using convex optimization.
- The download link for the HySpeFAS dataset will be available at the time of submitting the camera ready paper.

## Usage
### Preparation
- Install the requirements.
- Download the HySpeFAS dataset from the link provided in the Supplementary Material of the paper ().
- Unzip the dataset files and put them in the _data_ directory.
- (Optional) Run the script _reconstruction/reconstruct_images.m_ to get the reconstructed HSIs (R-HSIs). This is essential 
to train models based on R-HSI such as TwIST-ResNet.


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
