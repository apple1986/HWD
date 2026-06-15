# HWD: Haar Wavelet Downsampling for Semantic Segmentation

Official implementation of **Haar Wavelet Downsampling (HWD)** proposed in:

> **Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation**
> *Pattern Recognition, 2023*

HWD is a lightweight and plug-and-play downsampling module that decomposes feature maps into frequency components using the Haar wavelet transform, preserving more structural information than conventional downsampling operations. It can be directly used as a replacement for **Max Pooling**, **Average Pooling**, or **Stride Convolution** in existing semantic segmentation networks.

---

## 🚀 Installation

### 1. Install `pytorch_wavelets`

```bash
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```

For more details, please refer to:

https://github.com/fbcotter/pytorch_wavelets

---

### 2. Install `segmentation_models_pytorch`

```bash
pip install segmentation-models-pytorch
```

For more information, please visit:

https://github.com/qubvel/segmentation_models.pytorch

---

## 🔧 Usage

HWD is designed as a **drop-in replacement** for conventional downsampling operations.

Simply replace existing downsampling layers, such as:

* Max Pooling (`MaxPool2d`)
* Average Pooling (`AvgPool2d`)
* Strided Convolution (`Conv2d(stride=2)`)

with the proposed **HWD module**.

This allows existing segmentation architectures to benefit from frequency-aware feature extraction with minimal modifications.

---

## 📂 Datasets

### CamVid

* https://github.com/alexgkendall/SegNet-Tutorial

### Synapse

* Baidu Netdisk:
  https://pan.baidu.com/s/1PoCxb_BNAalm37pmPTOBmw

* Extraction code:

```text
asdf
```

---

## 📖 Citation

If you find this work useful for your research, please consider citing our paper.

```bibtex
@article{XU2023109819,
  title   = {Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation},
  author  = {Guoping Xu and Wentao Liao and Xuan Zhang and Chang Li and Xinwei He and Xinglong Wu},
  journal = {Pattern Recognition},
  volume  = {},
  pages   = {109819},
  year    = {2023},
  issn    = {0031-3203},
  doi     = {10.1016/j.patcog.2023.109819}
}
```

---

## 🔬 Related Works

If you are interested in wavelet-based representation learning or medical image segmentation, the following works may also be relevant:

1. **Xu, G., et al.** *Exploiting DINOv3-based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation.* Machine Learning: Science and Technology, 2026.

2. **Wu, X., et al.** *Wavelet Attention Fusion for Semi-Supervised Ultrasound Segmentation.* Biomedical Signal Processing and Control, 117 (2026): 109566.

3. **Yue, W., et al.** *Wavelet-Based Frequency Replacement and Edge Enhancement for Semi-Supervised Fetal Ultrasound Image Segmentation.* Journal of Ultrasound in Medicine, 2026.

4. **Xu, G., et al.** *Is the Medical Image Segmentation Problem Solved? A Survey of Current Developments and Future Directions.* arXiv:2508.20139, 2025.

---

## ⭐ Acknowledgement

This project builds upon the excellent open-source library **pytorch_wavelets**. We sincerely thank the authors for making their implementation publicly available.

---

## 📬 Contact

If you have any questions or suggestions, please feel free to open an issue or submit a pull request.

If this repository is helpful for your research, a ⭐ on GitHub would be greatly appreciated!
