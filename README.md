Complex-Valued Convolutional Neural Network with Learnable Activation Function for Frequency-Domain Radar Signal Processing
===========
This repository contains code which reproduces experiments presented in the paper [Complex-Valued Convolutional Neural Network with Learnable Activation Function for Frequency-Domain Radar Signal Processing]().
<p align="center">
	<img src="Architecture.png"  width="100%" height="100%">
	<br>
	<em>The Architecture of the proposed frequency-adaptive complex-valued convolutional neural network.</em>
</p>


## Requirements

The main requirements are listed below:

* Tested with NVIDIA RTX A6000 GPU 
* Tested with CUDA 11.8 and cuDNN 8700
* Python 3.12.3
* Ubuntu 20.04.6
* PyTorch 2.3.0+cu118
* Trainable parameters torch.complex64

## Datasets

* [DIAT-μRadHAR: Radar micro-Doppler Signature dataset for Human Suspicious Activity Recognition](https://ieee-dataport.org/documents/diat-mradhar-radar-micro-doppler-signature-dataset-human-suspicious-activity-recognition)
* [DIAT-µSAT: micro-Doppler Signature Dataset of Small Unmanned Aerial Vehicle (SUAV)](https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav)
* [MSTAR-10: Ten-Class Satellite Image Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)
* [EuroSAT: 13 Spectral Bands Sentinel-2 Satellite Image Datasett](https://github.com/phelber/eurosat)


## Steps for training and testing
To train and test the frequency domain complex-valued convolutional neural network, run the following commands:

* python HAR.py
* python SUAV.py
* python MSTAR-10.py
* python EuroSAT.py


If you find our work useful, you can cite our paper using:
```
@article{,
  publisher={IEEE}
}
```        
