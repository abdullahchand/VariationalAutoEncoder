import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.autoencoder import VAE
from PIL import ImageFile
from PIL import Image

"""

Plots the loss on valid dataset.

call :
        1 - "get_loss" with path to directory

"""
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = VAE()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

model.load_state_dict(torch.load("runs/model_at_best.pth"))
model.eval()


def get_loss(path_to_valid):
    """
    To evaluate the loss on valid dataset.

    Input:
        path_to_valid: Path to valid dataset contatining folder of images.
    
    returns:
        None
    
    prints:
        1 - Average Loss over the valid Dataset.
        2 - Plots the loss over the valid dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((254,254)),
        transforms.ToTensor()
    ])




    loss = 0
    count = 0
    loss_values = []
    for f in os.listdir(path_to_valid):
        for i in os.listdir(path_to_valid+"/"+f):
            image = Image.open(path_to_valid+"/"+f+"/"+str(i))
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            reconstructed_image, mean, log_var, z = model(image)
            
            # calculate the loss
            kl =  -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss =  F.binary_cross_entropy(reconstructed_image, image, reduction="sum")

            loss = kl + recon_loss
            loss_values.append(loss.item())
            loss+= loss
            count+=1

    
    print("Average Loss : " , loss.cpu().detach()/count)
    plt.plot(loss_values)
    plt.show()


get_loss("train-data/valid")