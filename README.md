# MSPNet: A Multi-Scale Pyramid Network for Semantic Segmentation of Urban-Scale Photogrammetric Point Clouds

This is the official implementation of **MSPNet: A Multi-Scale Pyramid Network for Semantic Segmentation of Urban-Scale Photogrammetric Point Clouds** <br />
Ziyin Zeng, Honglin Chen, Jian Zhou*, Bijun Li, and Ruili Wang. <br />

### (1) Setup
This code has been tested with Python 3.8, Tensorflow 2.4, CUDA 11.0 and cuDNN 8.0.5 on Ubuntu 20.04.
- Clone the repository 
```
git clone --depth=1 https://github.com/zeng-ziyin/MSPNet && cd MSPNet
```
- Setup python environment
```
conda create -n unext python=3.8
source activate unext
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Data Prepare
- Download HRHD-HK, UrbanBIS, SensatUrban, (optional) S3DIS, (optional) ScanNet
```
cd utils/
python data_prepare_$data_you_want_use$.py
```

### (3) Train & Test
```
python main_$data_you_want_use$.py --mode train & test
```
