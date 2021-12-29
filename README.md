# HistoGNN

Implementation of a Graph Neural Network for histology image classification tasks. 

## Table of Contents

1. [Installation](#installation)
2. [Requirements](#Requirements)
3. [Usage](#usage)

## Installation
HistoGNN has been developed using Python 3.7. Clone the repository:

```sh
git clone https://github.com/ChakraLab/HistoGNN
```

## Requirements
It depends on a few Python packages, including:
* dgl (0.6.1) - [dgl library](https://pypi.org/project/dgl-cu101/)
* torch (1.9.0+cu102)
* google_drive_downloader - [GoogleDriveDownloader](https://pypi.org/project/googledrivedownloader/)

## Usage
The code can be simply run with:

```sh
python cgnn_run.py
```

The optional arguments are:

```bash
usage: cgnn_run.py [--gpus GPUS] [--nodes NODES]

Arguments:
  --gpu GPU             gpu (-1 for no GPU, 0 otherwise)
  --gpus                number of GPUs
  --nodes               number of nodes
