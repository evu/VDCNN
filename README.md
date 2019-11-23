# VDCNN #
*Tensorflow 2.0 Implementation of Very Deep Convolutional Neural Network for Text Classification.*

## Note ##
This repository is a simple Tensorflow 2.0 implementation of the VDCNN model proposed by Conneau et al. in [their 2016 paper](https://arxiv.org/abs/1606.01781). It is based off of [this Keras implementation by zonetrooper32](https://github.com/zonetrooper32/VDCNN).

Note: Temporal batch norm has not been implemented. *"Temp batch norm applies same kind of regularization as batch norm, except that the activations in a mini-batch are jointly normalized over temporal instead of spatial locations."* Right now, this project is using Tensorflow's standard batch normalization.

It should be noted that the original implementation by the authors of the VDCNN paper was done in Touch 7.

## Prerequisites ##

 - Python 3
 - Tensorflow 2.0
 - numpy

## Datasets ##

The original paper tests several NLP datasets, including DBPedia, AG's News, Sogou News, etc. [`data_loader.py`](data_loader.py) expects CSV-formatted train and test files.

Downloads of those NLP text classification datasets can be found here (Many thanks to ArdalanM):

| Dataset                | Classes | Train samples | Test samples | source |
|------------------------|:---------:|:---------------:|:--------------:|:--------:|
| AG's News              |    4    |    120 000    |     7 600    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Sogou News             |    5    |    450 000    |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| DBPedia                |    14   |    560 000    |    70 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Full       |    5    |    650 000    |    50 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|

A script to generate GloVe vector embeddings from the *AG's News* dataset is located at [`scripts/txt2embedding.py`](scripts/txt2embedding.py). It has its own dependencies that are independent of the main project, located in the `script_requirements.txt` file in the same folder.

Usage: 

## Hardware ##

Training and testing were performed on an Ubuntu 16.04 server with an NVIDIA Quadro GP100, using the configuration defaults defined in [`train.py`](train.py). The dataset used was the AG's News dataset.

## References ##

[Keras implementation by zonetrooper32](https://github.com/zonetrooper32/VDCNN)

[Original preprocessing codes and VDCNN Implementation By geduo15](https://github.com/geduo15/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow)

[Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

[NLP Datasets Gathered by ArdalanM and Others](https://github.com/ArdalanM/nlp-benchmarks)
