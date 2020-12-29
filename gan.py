# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 19:06:56 2020

@author: Fra
"""
import torch
import torchvision
from torch import nn
import math
from matplotlib import pyplot as plt
import cv2 as cv

torch.manual_seed(111)

train_data_length = 1024
train_data = torch.zeros((train_data_length,2))
train_data[:, 0] = torch.rand(train_data_length) * 2 * math.pi
train_data[:, 1] = torch.sin(train_data[:, 0])
#print(train_data.shape)
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

#plt.plot(train_data[:, 0].detach().numpy(), train_data[:, 1].detach().numpy(), ".")

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
            )
        
    def forward(self,x):
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            )
        
    def forward(self,x):
        output = self.model(x)
        return output
    
discriminator = Discriminator()
generator = Generator()

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        real_samples_labels = torch.ones((batch_size,1))
        latent_space_samples = torch.randn((batch_size,2))
        
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size,1))
        
        all_samples = torch.cat((real_samples,generated_samples))
        all_samples_labels = torch.cat((real_samples_labels,generated_samples_labels))
        
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator,all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        latent_space_samples = torch.randn((batch_size, 2))

        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()
        
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            #plt.plot(generated_samples[:, 0].detach().numpy(), generated_samples[:, 1].detach().numpy(), ".")
            #plt.show()



latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

plt.plot(generated_samples[:, 0].detach().numpy(), generated_samples[:, 1].detach().numpy(), ".")











































