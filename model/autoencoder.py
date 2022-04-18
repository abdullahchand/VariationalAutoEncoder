import torch.nn as nn
import torch.nn.functional as F
import torch
# define the NN architecture
class encoder(nn.Module):
    def __init__(self, z_dim = 512):
        super(encoder, self).__init__()
        """
        Encoder Network.
        Inputs:
            1 - z_dim is the size of the latent space we want
        
        Returns:
            1 - The output after passing through the encoder.
            2 - Mu is the mean.
            3 - log_var is the log variance. 
        """
        #Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)  
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)


        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)  
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
       
        self.conv5 = nn.Conv2d(128, 128, 3, stride=2)  
        self.conv6 = nn.Conv2d(128, 256, 3, stride=2)

        self.conv7 = nn.Conv2d(256, 256, 1)


        self.mu = nn.Linear(1024, z_dim)
        self.log_var = nn.Linear(1024, z_dim) 


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x)) #4x4 output
        x = F.relu(self.conv7(x))
        x = x.view(x.size(0), -1)
        mu, log_var = self.mu(x),self.log_var(x)

        return x, mu, log_var



class decoder(nn.Module):
    def __init__(self, z_dim=512):
        super(decoder, self).__init__()
        """
        Decoder Network.

        Inputs:
            1 - z_dim is the size of the latent space we want
        
        Returns:
            1 - The output after passing through the decoder network.
        """
        #Decoder
        self.fc = nn.Linear(z_dim, 1024)
        self.t_conv0 = nn.ConvTranspose2d(256, 256, 1)
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 4, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 128, 4, stride=2)
        
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        
        self.t_conv5 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.t_conv6 = nn.ConvTranspose2d(32, 3, 4, stride=2)

        
        

    def unflatten(self, input, size=256):
        return input.view(input.size(0), size, 2, 2)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = F.relu(self.t_conv0(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        x = F.sigmoid(self.t_conv6(x)) #254x254 output
        return x


class VAE(nn.Module):
    """
        Complete variational autoencoder network. Takes the output from the encoder netwrok, samples from the normal distribution given the mean and log variance. Passes that to the decoder network
        to train end to end. The variational autoencoder works by sampling from the space in which the encoder networks output lies, this ensures that the model learns to keep the latent space in some distribution.
        Returns:
            1 - The output after passing through the encoder.
            2 - Mu is the mean.
            3 - log_var is the log variance. 
        """
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = encoder()
        self.decoder = decoder()

    def sample_from_distribution(self,mu,logvar):
        """
        Samples from a normal distribution, given the mean and variance to construct the distribution.
        Inputs:
            1 - mu is the mean value from the encoder network output.
            2 - log_var is the log variance from the encoder network output.
        
        Returns:
            1 - The output after passing through the encoder.
            2 - Mu is the mean.
            3 - log_var is the log variance. 
        """
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two

        sample = torch.randn_like(std)

        return sample * std + mu

    def representation(self, x):
        """
        returns the latent space from the model
        Inputs:
            1 - x is the image input to the network
        
        Returns:
            1 - returns a sample from the learned latent space distribution.
        """
        x, mu, log_var = self.encoder(x)
        return self.sample_from_distribution(mu,log_var)


    def forward(self, x):
        x, mu, log_var = self.encoder(x)

        z = self.sample_from_distribution(mu,log_var)

        x = self.decoder(z)

        return x, mu, log_var, z
