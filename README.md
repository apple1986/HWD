# HWD
A Simple but Effective Downsampling Module For Semantic Segmentation

1. install pytorch_wavelets
`
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
`
Please refer for more details: https://github.com/fbcotter/pytorch_wavelets

2. install semantic segmentation package 
`
pip install segmentation-models-pytorch
`
Please refer for more details: https://github.com/qubvel/segmentation_models.pytorch

4. You can replace the downsample method, like maxpooling, averagepooling, convolution with stride, with our HWD directly.

5. Dataset
(1) Camvid: https://github.com/alexgkendall/SegNet-Tutorial
(2) Synapse: https://pan.baidu.com/s/1PoCxb_BNAalm37pmPTOBmw   Code: asdf

Please cite our paper if it is helpful to you. Thanks.
`
@article{XU2023109819,
title = {Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation},
journal = {Pattern Recognition},
pages = {109819},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109819},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323005174},
author = {Guoping Xu and Wentao Liao and Xuan Zhang and Chang Li and Xinwei He and Xinglong Wu},
keywords = {Semantic segmentation, Downsampling, Haar wavelet, Information Entropy}
}
`
