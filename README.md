# FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models

Dataset abd code for the paper *FMARS: Annotating Remote Sensing Images for Disaster Management using Foundation Models*.

[![arXiv](https://img.shields.io/badge/arXiv-2210.10732-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2210.10732)

## Environment Setup

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

## Installation and Use

> [!WARNING]
> TODO

## Acknowledgements

FMARS is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
