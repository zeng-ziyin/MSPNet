# MSPNet: A Multi-Scale Pyramid Network for Semantic Segmentation of Urban-Scale Photogrammetric Point Clouds

### (1) Setup
This code has been tested with Python 3.8, Tensorflow 2.4, CUDA 11.0 and cuDNN 8.0.5 on Ubuntu 20.04.

- Setup python environment
```
conda create -n unext python=3.8
source activate unext
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Data Prepare
```
cd utils/
python data_prepare_$data_you_want_use$.py
```

### (3) Train & Test
```
python main_$data_you_want_use$.py --mode train & test
```
