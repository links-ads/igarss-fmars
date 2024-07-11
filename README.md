# FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models

Code for the experiments from the paper *FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models*.

![Samples](resources/qualitatives-02.png)
*Qualitative results obtained over two example areas, namely USA (top) and Gambia (bottom). from left to right: RGB image, DAFormer, MIC, and FMARS ground truth. Best viewed zoomed in.*

[![arXiv](https://img.shields.io/badge/arXiv-2405.20109-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.20109)

> [!NOTE]  
> The Dataset is available as [parquet files](https://huggingface.co/datasets/links-ads/fmars-dataset), to be rasterized, or can be generated with the package [igarss-fmars-gen](https://github.com/links-ads/igarss-fmars-gen).


## Installation

### Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
pyenv install 3.8.17
pyenv global 3.8.17
python -m venv .venv
source .venv/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -e .
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

### Metadata 

Download necessary metadata using:
```shell
wget -q https://github.com/links-ads/igarss-fmars-gen/releases/download/v0.1.0/metadata.zip
unzip metadata.zip
rm metadata.zip
```

### Dataset generation

The Dataset is available as [parquet files](https://huggingface.co/datasets/links-ads/fmars-dataset), to be rasterized, or can be generated with the package [igarss-fmars-gen](https://github.com/links-ads/igarss-fmars-gen) Both the original Maxar Open Data dataset and the annotations genereted with FMARS are needed. An example of data folder structure can be:

```
data/
├── maxar-open-data/
|   ├── afghanistan-earthquake22/
|   ├── BayofBengal-Cyclone-Mocha-May-23/
|   └── ...
└── annotations/
    ├── train/
    |   ├── afghanistan-earthquake22/
    |   ├── BayofBengal-Cyclone-Mocha-May-23/
    |   └── ...	
    ├── val/
    |   ├── afghanistan-earthquake22/
    |   ├── BayofBengal-Cyclone-Mocha-May-23/
    |   └── ...	
    └── test/
        ├── afghanistan-earthquake22/
        ├── BayofBengal-Cyclone-Mocha-May-23/
        └── ...	
``` 

It's necessary for the images and the annotations to be in the correct events subfolders. The paths need to be specified in the configuration files under ``` configs/_base_/datasets/``` in the fields ```img_dirs``` and ```ann_dirs```.

## Experiments

To run the experiments reported in the paper, the following commands are used to train and test the three models (SegFormer, DAFormer, MIC).

```
python -B -O tools/train.py configs/fmars/segformer.py
python -B -O tools/test.py configs/fmars/segformer.py work_dirs/segformer/iter_30000.pth --eval mIoU --show-dir output_images/segformer

python -B -O tools/train.py configs/fmars/daformer_sampler.py
python -B -O tools/test.py configs/fmars/daformer_sampler.py work_dirs/daformer_sampler/iter_30000.pth --eval mIoU --show-dir output_images/daformer_sampler

python -B -O tools/train.py configs/fmars/mic_sampler.py
python -B -O tools/test.py configs/fmars/mic_sampler.py work_dirs/mic_sampler/iter_30000.pth --eval mIoU --show-dir output_images/mic_sampler
```

Additionally, in the ```configs/``` folder, alternative experiments are available, including the versions of the models trained ignoring the entropy based sampling.

## Acknowledgements

FMARS is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
