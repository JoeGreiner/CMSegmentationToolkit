
# CMSegmentationToolkit
![GithubSegmentationToolkit](https://github.com/user-attachments/assets/07c53edf-d60d-4103-b80d-555c5aa7708d)


This repository contains code for the paper 'A Deep Learning-Enabled Toolkit for the Reconstruction of Ventricular Cardiomyocytes' (under review). We provide a code and GUIs for the deep learning-based restoration and segmentation of 3D confocal microscopy data of WGA-labeled ventricular tissue. We recommend proofreading the resulting segmentations with [SegmentPuzzler](https://github.com/JoeGreiner/SegmentPuzzler). In case your image data differs a lot from ours or you're not happy with the performance, retraining the models on your data is likely a good idea.

## Installation

Requirements: Linux, Windows (tested on Ubuntu 24.04; Windows 11 Pro 23H2). Modern PyTorch-compatible graphic card (tested on NVIDIA 4090; 3090, most likely an NVIDIA card >10GB RAM should suffice).

<details>
<summary>Windows</summary>

There are two environments available: one for PyTorch GPU/TensorFlow CPU, and another for TensorFlow GPU only. TensorFlow is required for the restoration workflow. On Windows, TensorFlow GPU works only with versions <2.11 and cudnn 8.1.0/cuda 11.2, which are incompatible with the latest PyTorch versions. You can switch environments to use TensorFlow (GPU) for restoration, taking advantage of GPU acceleration. Alternatively, you can use the conda_env_windows.yml, which runs the restoration workflow on the CPU. Despite being slower, it handles small/medium stacks in a reasonable time due to the network's size. If unsure, please use conda_env_windows.yml.

Steps:
1. Clone/download this repository and navigate to the folder.
``` bash
git clone https://github.com/JoeGreiner/CMSegmentationToolkit.git
cd CMSegmentationToolkit
```
2. Install the conda environment.
```
 conda env create --file conda_env_windows.yml
OR
 conda env create --file environment_windows_tf_GPU.yml  (for tensorflow/GPU)
```
3. Activate the conda environment.
```
conda activate CMSegmentationToolkit
OR
conda activate CMSegmentationToolkitTF (for tensorflow/GPU)
```
4. Install the package. 
```
pip install .
```
5. Use the GUIs for restoration/ segmentation.
```
python A_restoration_GUI.py
python B_segmentation_GUI.py
python C_analyse_morphology_GUI.py
```
</details>


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
python C_analyse_morphology_GUI.py
```
</details>


## GUIs
<details>
<summary>DTWS-MC Cardiomyocyte Segmentation</summary>

https://github.com/user-attachments/assets/7fdaae7f-f879-4341-b6d8-7c20df6e2f9b

(Please be aware that DTWS-MC may take some while until the processing is finished. In the demo, we run without test time augmentation and ensembling, which accelerates the segmentation. Proofreading at the end with SegmentPuzzler is optional, but recommended.)
</details>

<details>
<summary>WGA Image Restoration with CARE</summary>

 
https://github.com/user-attachments/assets/1fb99faf-d9a8-4085-bf19-a24650fda5fc


</details>
<details>
<summary>Cardiomyocyte Morphology Analysis</summary>

 
https://github.com/user-attachments/assets/bb4d4d24-30d3-44b8-b074-9a7c400e759f


(Opening in SegmentPuzzler is just to visualise the loaded data, but it's not necessary for the workflow.)
</details>

<details>
<summary>Cardiomyocyte Distance To Nearest TATS Analysis</summary>


https://github.com/user-attachments/assets/7ddcda0e-d019-42c8-8577-41f938c0ea19


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
