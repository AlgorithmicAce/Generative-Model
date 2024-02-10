# Generative Model

This repository will contain codes for Generative Models. Generative Models include ***Autoencoder***, ***Restricted Boltzmann Machines (RBM)*** and ***Generative Advesarial Networks (GAN)***

**Autoencoders** can be used for the following usages and much more:
1.  Detecting Anomalies from data
2.  Removing Noise from data
3.  Image Colorization

I have attached a basic Convolutional Autoencoder and Noise Remover using Autoencoder. You may clone the repository and try playing with it.

##### Noise Remover using Autoencoder

This Autoencoder trains on normal data and then tries to reconstruct it and then undergoes the backpropagation method. This allows the model to capture the important features from the data **(FEATURE EXTRACTION)** Then, when the data with noise is entered (input), the Autoencoder extracts the important features an filters out the noises. 

**Generative Advesarial Networks** uses **2 different** networks, which are **generative & discriminative** network. If the generative network can create a data that districiminative network classifies as true data, then that data is generated and stored.

