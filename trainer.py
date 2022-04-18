import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.autoencoder import VAE
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
To train the VAE model on the given dataset.

run :
        1 - Python trainer.py

stores:
        1 - The trained models in dir -> runs/
"""



BATCH_SIZE = 16

# Load Model

model = VAE()

print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# transform : RandResizedCrop -> So that we can have some augmentation to the dataset, ToTensor -> Convert to tensor
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=254, scale=(0.2, 1.)),
    transforms.ToTensor()
 ])

# No need to create custom class. We will just use the ImageFolder class for data class.
train_data = datasets.ImageFolder('train-data/train', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=4, shuffle = True)

# Epochs -> full size of train_data / Batch Size
n_epochs = len(train_data)//BATCH_SIZE

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_values = []

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    c = 0
    for data in train_loader:
        c+=BATCH_SIZE
        print("Step : ", (len(train_data)-c)/BATCH_SIZE)
        # _ stands in for labels, here
        images, _ = data
        images = images.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        reconstructed_image, mean, log_var, z = model(images)

        # calculate the loss
        # We define the KL divergence loss. It essentially makes sure that the latent space values stay between the given (Normal) Distribution.
        kl =  -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # For recconstruction loss we use BCELoss. It can be replaced by MSE or Monte Carlo Elbo loss
        recon_loss = F.binary_cross_entropy(reconstructed_image, images, reduction="sum")

        loss = kl + recon_loss
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
    
    # At every 5 epochs, Save th model.
    if epoch%5 == 0:
        torch.save(model.state_dict(),"runs/model_at_" + str(epoch)+ "_best.pth")

    #append to loss_values so we can plot the loss later
    loss_values.append(train_loss/len(train_loader))

            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))


torch.save(model.state_dict(),"runs/model_at_best.pth")
plt.plot(loss_values)
plt.show()