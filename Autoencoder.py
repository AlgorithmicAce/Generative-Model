import torch
from torch import nn, optim
from torchvision import datasets as dsets
from torchvision import transforms as tforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

mnist_data = dsets.MNIST(root = './data', train = True, download = True, transform = tforms.ToTensor())

mnist_load = DataLoader(dataset = mnist_data, batch_size = 500)

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = LinearAutoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 10

for epoch in range(n_epochs):
    for images, label in mnist_load:
        images = images.view(-1, 784)
        recon = model(images)
        loss = criterion(images, recon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", str(epoch + 1), "Loss:", loss.item())

for i in range(10):
  a = mnist_data[i][0]
  b = a.view(-1, 784)
  c = model(b)
  d = c.view(28,28)
  a = a.view(28,28)
  d = d.detach()
  _, image = plt.subplots(1,2)
  image[0].imshow(a, cmap = 'gray')
  image[1].imshow(d, cmap = 'gray')
  plt.show()