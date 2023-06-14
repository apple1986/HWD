# HWD
A Simple but Effective Downsampling Module For Semantic Segmentation

1. install pytorch_wavelets
`
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
`
Please ref for more details: https://github.com/fbcotter/pytorch_wavelets

2. install semantic segmentation package 
`
pip install segmentation-models-pytorch
`
Please refer for more details: https://github.com/qubvel/segmentation_models.pytorch

4. You can replace the downsample method, like maxpooling, averagepooling, convolution with stride, with our HWD directly.

5. Dataset
(1) Camvid: https://github.com/alexgkendall/SegNet-Tutorial
(2) Synapse: https://pan.baidu.com/s/1PoCxb_BNAalm37pmPTOBmw   Code: asdf
