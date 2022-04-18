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
import random

"""
To Plot the Cosine similarity distribution of the latent space.

call :
        1 - "same_image_crop" with path to data directory, to plot the same image crops cosine similarity.
        2 - "different_image_crop" with path to data directory, to plot the different image crops cosine similarity.
"""


ImageFile.LOAD_TRUNCATED_IMAGES = True

model = VAE()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

model.load_state_dict(torch.load("runs/model_at_best.pth"))
model.eval()


transform = transforms.Compose([
        transforms.RandomResizedCrop(size=254, scale=(0.2, 1.)),
        transforms.ToTensor()
    ])

def get_latent_vector(path_to_image):
    """
    Input:
        path_to_image: Path to image.
    
    returns:
        1 - Latent space/feature vector from the model.
    
    prints:
        None
    """

    image = Image.open(path_to_image)
    image = transform(image)

    image = image.unsqueeze(0)
    image = image.to(device)


    z_1 = model.representation(image)

    return z_1


def same_image_crop(path_to_dir):
    """
    Input:
        path_to_dir: Path to data directory containing folder of images.
    
    returns:
        None
    
    prints:
        1 - Plots the cosine similarity of the same image crops
    """


    cos_values = []
    for f in os.listdir(path_to_dir):
        for i in os.listdir(path_to_dir+"/"+f):
            image = Image.open(path_to_dir+"/"+f+"/"+str(i))
            image_crop_1 = transform(image)
            image_crop_2 = transform(image)

            image_crop_1 = image_crop_1.unsqueeze(0)
            image_crop_1 = image_crop_1.to(device)

            
            image_crop_2 = image_crop_2.unsqueeze(0)
            image_crop_2 = image_crop_2.to(device)
            
            z_1 = model.representation(image_crop_1)
            z_2 = model.representation(image_crop_2)

            cos = nn.CosineSimilarity()

            # gets the cosine similarity and appends it to the list f cos_values.
            cos_values.append(cos(z_1,z_2).item())


    plt.hist(cos_values)
    plt.show()   

def different_image_crop(path_to_dir):

    """
    Input:
        path_to_dir: Path to data directory containing folder of images.
    
    returns:
        None
    
    prints:
        1 - Plots the cosine similarity of the different image crops
    """

    cos_values = []
    for f in os.listdir(path_to_dir):
        choice1 = random.choice(os.listdir(path_to_dir))
        z_1 = None
        z_2 = None

        for i in os.listdir(path_to_dir+"/"+choice1):
            z_1 = get_latent_vector(path_to_dir+"/"+choice1+"/"+str(i))

        choice2 = random.choice(os.listdir(path_to_dir))
        for i in os.listdir(path_to_dir+"/"+choice2):
            z_2 = get_latent_vector(path_to_dir+"/"+choice2+"/"+str(i))
        
        
        cos = nn.CosineSimilarity()

        cos_values.append(cos(z_1,z_2).item())


    plt.hist(cos_values)
    plt.show()       

print("Getting Similarity of Same Images...")
same_image_crop("train-data/valid")
print("Getting Similarity of Different Images..")
different_image_crop("train-data/valid")