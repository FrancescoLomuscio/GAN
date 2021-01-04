# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 22:36:29 2021

@author: Fra
"""
import torch
from torch import nn

import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

real_samples , mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4,4,i+1)
    plt.imshow(real_samples[i].reshape(28,28),cmap = "gray_r")
    plt.xticks([])
    plt.yticks([])

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
            )
        
    def forward(self,x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784)
            )
        
    def forward(self,x):
        output = self.model(x)
        output = output.view(x.size(0),1,28,28)
        return output
    
discriminator = Discriminator()
generator = Generator()

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        real_samples_labels = torch.ones((batch_size,1))
        latent_space_samples = torch.randn((batch_size,100))
        
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size,1))
        
        all_samples = torch.cat((real_samples,generated_samples))
        all_samples_labels = torch.cat((real_samples_labels,generated_samples_labels))
        
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator,all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        latent_space_samples = torch.randn((batch_size, 100))

        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()
        
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            #plt.plot(generated_samples[:, 0].detach().numpy(), generated_samples[:, 1].detach().numpy(), ".")
            #plt.show()



latent_space_samples = torch.randn(batch_size, 100)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
