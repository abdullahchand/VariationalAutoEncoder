# VariationalAutoEncoder
This is specifically for Quixel Test.



Start with trainer.py.

<h1>Training</h1>

1 - Clean the dataset from all the json files. <br>
2 - Split data into train and valid. <br>
3 - Place the dataset into the root directory with the root folder of the dataset as "train-data". <br>

Then run
```
pip install -r requirments.txt

python trainer.py. <br>

```


<h1> Model </h1>
The Model architecture contains 7 convolutional layers. 
<h2> Encoder </h2>
. The Encoder network consists of 6 3x3 convolutional layer followed by 1x1 convolutional layer. <br>
. Batch Normalization is performed at convolutional layer 2 and convolutional layer 4. <br>


The following snipper shows the encoder architecture. <br>

```
encoder(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv5): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
  (conv6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
  (conv7): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (mu): Linear(in_features=1024, out_features=512, bias=True)
  (log_var): Linear(in_features=1024, out_features=512, bias=True)
)
```
<h2> Decoder </h2>
The Decoder network is symmetric to the encoder network, containing 1 1x1 transposed convolutional layer followed by 6 3x3 transposed Covloutional layer.<br>
  
The following snipped shows the decoder network. <br>
  ```
  decoder(
  (fc): Linear(in_features=512, out_features=1024, bias=True)
  (t_conv0): ConvTranspose2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (t_conv1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2))
  (t_conv2): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2))
  (t_conv3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))
  (t_conv4): ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(2, 2))
  (t_conv5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2))
  (t_conv6): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2))
)
  ```
  <h2> VAE </h2>
  The Vae network consists of the combination of the encoder and the decoder networ. It has additional functions to get random sample from the distribution and to get the latent representation given an image. <br>
  
  The following snippet shows the full network. <br>
  ```
  VAE(
  (encoder): encoder(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
    (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv5): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
    (conv6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
    (conv7): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (mu): Linear(in_features=1024, out_features=512, bias=True)
    (log_var): Linear(in_features=1024, out_features=512, bias=True)
  )
  (decoder): decoder(
    (fc): Linear(in_features=512, out_features=1024, bias=True)
    (t_conv0): ConvTranspose2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (t_conv1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2))
    (t_conv2): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2))
    (t_conv3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))
    (t_conv4): ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(2, 2))
    (t_conv5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2))
    (t_conv6): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2))
  )
)
```


<h1> Loss </h1>
We use the standard KL divergence loss (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) + binary cross entropy loss for reconstruction loss. <br>
We can use MSE Loss for reconstruction loss but through testing we found that binary cross entropy works better.<br>
The Loss values (training) goes as follow:- <br>

![Loss On Train](https://user-images.githubusercontent.com/30110060/163776568-d14d4526-65e4-4eef-b4a3-712b0d37fa1d.png)

we can see the loss after 16 epochs is : 1758306<br>

<h2> Loss on valid </h2>
See and run <br>
```
python AEtest.py
```

to plot the loss value on train or valid dataset. <br>

Our loss on valid data set comes out to be :- <br>

![Loss On Valid](https://user-images.githubusercontent.com/30110060/163774945-7760175f-ed75-4e18-bb3a-af1eec13b573.png)



<h1> Feature Similarity </h1>

See the file feature_similarity.py. In the file there are two functions, 1. To plot cosine similarity from same images. 2. To plot cosine similarity from different images. <br>

Run <br>
```
python feature_similarity.py
```

We use the cosine similarity to measure similarity between same images latent space and different images latent space.<br>
We first get random crops from the images and then measure the cosine similarity from crops taken from the same image and crops taken from the different images. <br>
We plot a histogram to show the results of where the majority lies. <br>
<h2> Same Image </h2>

![same-image](https://user-images.githubusercontent.com/30110060/163775229-ed7b210b-b5b6-4ede-894a-472ea785ad04.png)

We can see that in same image, there is a shift from 0 to 1, where maximum value close to 0.45.<br>

<h2> Different Image </h2>

![different-image](https://user-images.githubusercontent.com/30110060/163775407-e16c53f7-e4f0-418f-93d0-107b99e5a3d3.png)

We can see that in different image crops, there is a shift from 0.075 to -0.125. A dissimilarity is found in different Images. <br>


