
# CMSegmentationToolkit
![GithubSegmentationToolkit](https://github.com/user-attachments/assets/07c53edf-d60d-4103-b80d-555c5aa7708d)


This repository contains code for the paper 'A Deep Learning-Enabled Toolkit for the Reconstruction of Ventricular Cardiomyocytes' (under review). We provide a code and GUIs for the deep learning-based restoration and segmentation of 3D confocal microscopy data of WGA-labeled ventricular tissue. We recommend proofreading resulting segmentations with [SegmentPuzzler](https://github.com/JoeGreiner/SegmentPuzzler).

## Installation

Requirements: Linux (tested on Ubuntu 24.04). Modern PyTorch-compatible graphic card (tested on NVIDIA 4090; most likely a NVIDIA card >10GB RAM should suffice).

<details>
<summary>Linux</summary>

Steps:
1. Clone/download this repository and navigate to the folder.
``` bash
git clone https://github.com/JoeGreiner/CMSegmentationToolkit.git
cd CMSegmentationToolkit
```
2. Install the conda environment.
```
 conda env create --file environment_linux.yml
```
3. Activate the conda environment.
```
conda activate CMSegmentationToolkit
```
4. Install the package. 
```
pip install .
```
5. Use the GUIs for restoration/ segmentation.
```
python A_restoration_GUI.py
python B_segmentation_GUI.py
```
</details>

## Acknowledgements
We are further really thankful for the fantastic software packages we build upon. These include, but are not limited to, the following packages -- thank you! :
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* [elf](https://github.com/constantinpape/elf)
* [plant-seg](https://github.com/kreshuklab/plant-seg)
* [LUCYD](https://github.com/ctom2/lucyd-deconvolution)
* [stardist](https://github.com/stardist/stardist)
* [Napari](https://napari.org/stable/)
* [(py)clesperanto(_prototype)](https://github.com/clEsperanto/pyclesperanto_prototype)
* [ImageJ/Fiji](https://fiji.sc/)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [ITK](https://itk.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
