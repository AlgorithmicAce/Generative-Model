#Importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn, optim
import torchvision.datasets as dsets
import torchvision.transforms as tforms
from torch.utils.data import DataLoader as DL

#Uncomment all lines of code if you have CUDA supported GPU
#device = torch.device('cuda:0')

#Loading the MNIST Dataset
mnist_data = dsets.MNIST(root = './data', download = True, train = True, transform = tforms.ToTensor())
mnist_load = DL(dataset = mnist_data, batch_size = 500)

#Creating Convolutional Autencoder Class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

#Creating an object
model = Autoencoder()
#model.to(device)

#Configuring the loss function and optimizer function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training the model
for i in range(10):
  for images_, label_ in mnist_load:
    #images_, label_ = images.to(device), label.to(device)
    recon = model(images_)
    loss = criterion(images_, recon)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print("Epoch:", str(i + 1), "Loss:", loss.item())


#Adding noises to images
for i in range(10):
  aori = mnist_data[i][0]
  noise_factor = 0.2
  anoise = aori + noise_factor * torch.randn(aori.size())
  torch.clamp(anoise, 0, 1)
  aori = aori.view(-1, 28,28)
  anoise = anoise.view(-1, 28, 28)
  #anoise = anoise.to(device)
  arecon1 = model(anoise)
  #arecon2 = model(arecon1)
  anoise = anoise.detach()
  arecon1 = arecon1.detach()
  #arecon2 = arecon2.detach()
  #anoise = anoise.cpu()
  #arecon = arecon.cpu()
  _, image = plt.subplots(1,3)
  image[0].imshow(anoise.squeeze(), cmap = 'gray')
  image[1].imshow(arecon1.squeeze(), cmap = 'gray')
  #image[2].imshow(arecon2.squeeze(), cmap = 'gray')
  image[2].imshow(aori.squeeze(), cmap = 'gray')
  lossrecon = criterion(arecon1, aori)
  print("The loss between the reconstructed image and the original image:", lossrecon.item())
  plt.show()