# CNN for analysis and classification of tumor images 
## Overview
PyTorch-based project to deploy several convolutional neural networks predict a tumour’s mutational status. 
Models have been trained on publicly available dataset and showed final accuracy > 99% with <1% of false negatives.

## Description:
1. cnn-models.py - PyTorch script with scalable 2D and 3D CNNs for binary classification;
2. /data - tumor image patches and corresponding labels (mutation status)
3. tumor-utils.py - utility function for data loading and transformation and other utility functions
4. 1-tumor-classifier-2d.py and 2-tumor-classifier-3d.py  - scrips deploying correspondingly 2D and 3D networks;
5. /reports -  graphs showing typical learning curves
6. tech-notes.txt  - notes on model performance during testing and hyperparameters tuning.
