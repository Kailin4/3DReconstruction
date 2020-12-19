!pip install pywavefront
!pip install pyglet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from skimage.transform import resize
import cv2
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import matplotlib
import torchvision
import pywavefront
import plotly.graph_objects as go

import torchvision.utils as utils 

torch.manual_seed(1000)

class ImageDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features1 = self.features[index]
        label1 = self.labels[index]
        return features1, label1


def load_data(batch_s, label_train, train, label_val, val, label_test, test):
    train_file = ImageDataset(train, label_train)
    val_file = ImageDataset(val, label_val)
    test_file = ImageDataset(test, label_test)
    train_loader = torch.utils.data.DataLoader(train_file, batch_size=batch_s, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_file, batch_size=batch_s, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_file, batch_size=batch_s, shuffle=True)
    return train_loader, val_loader, test_loader


def load_model(lr1):
    model1 = ModelA()
    optim1 = optim.Adam(model1.parameters(), lr=lr1, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    return model1, optim1, criterion


def training(batch_size, lr, epochs, enc_weights, dec_weights, cuda1, train_loader):
  
    torch.manual_seed(1001)
    sample_fac = 4
    model1, optimizer, criterion = load_model(lr)

    # If we want to continue training from where we left off:
    if enc_weights != '':
        model1.load_state_dict(torch.load(enc_weights))

    train_loss = []
    valid_loss = []
    num_samples_trained = 0

    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        model1.train()
        for i, batch in enumerate(train_loader):
            data, label = batch
            num_samples_trained += batch_size
            #print("data shape", data.shape) # torch.Size([16, 2, 1200, 1600, 3])
            data1 = data.reshape([data.shape[0], data.shape[-1], data.shape[1], data.shape[2]])
            # Training 
            optimizer.zero_grad()
            reconstruction, mu, logvar = model1(data1.float())
            #print(reconstruction.detach().numpy().shape) # (4, 3, 1200, 1600)
            plt.imshow(reconstruction.reshape([reconstruction.shape[0], reconstruction.shape[2], reconstruction.shape[3], reconstruction.shape[1]]).detach().numpy()[0]), plt.show()
            loss1 = criterion(reconstruction, label)
            loss = final_loss(loss1, mu, logvar)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            break
        train_epoch_loss = float(running_loss/num_samples_trained)
        train_loss.append(train_epoch_loss)
        val_loss = validate(model1)
        valid_loss.append(val_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")
    epochs_arr = np.arange(epochs)
    plt.plot(epochs_arr, train_loss)
    plt.plot(epochs_arr, valid_loss)
    return True


class ModelA_simple(nn.Module):
    def __init__(self):
        super(ModelA_simple, self).__init__()
 
        self.kernel_size = 6
        self.init_kernel = 16

        self.enc_1 = nn.Conv2d(in_channels=3, out_channels=self.init_kernel, kernel_size=self.kernel_size)
        self.enc_2 = nn.Conv2d(in_channels=self.init_kernel, out_channels=self.init_kernel*2, kernel_size=self.kernel_size)
        self.enc_3 = nn.Conv2d(in_channels=self.init_kernel*2, out_channels=self.init_kernel*4, kernel_size=self.kernel_size)
        self.enc_4 = nn.Conv2d(in_channels=self.init_kernel*4, out_channels=self.init_kernel*8, kernel_size=self.kernel_size)
        self.enc_5 = nn.Conv2d(in_channels=self.init_kernel*8, out_channels=self.init_kernel*2, kernel_size=self.kernel_size)

        self.dec_1 = nn.ConvTranspose2d(in_channels=self.init_kernel*2, out_channels=self.init_kernel*4, kernel_size=self.kernel_size)
        self.dec_2 = nn.ConvTranspose2d(in_channels=self.init_kernel*4, out_channels=self.init_kernel*8, kernel_size=self.kernel_size)
        self.dec_3 = nn.ConvTranspose2d(in_channels=self.init_kernel*8, out_channels=self.init_kernel*8, kernel_size=self.kernel_size)
        self.dec_4 = nn.ConvTranspose2d(in_channels=self.init_kernel*8, out_channels=self.init_kernel*15, kernel_size=self.kernel_size)
        self.dec_5 = nn.ConvTranspose2d(in_channels=self.init_kernel*15, out_channels=self.init_kernel*22, kernel_size=self.kernel_size)
    
    def shuffling(self, a1, b1):
        s_d = torch.exp(0.5*b1) 
        s_2 = torch.randn_like(s_d)
        val1 = a1 + (s_2 * s_d) 
        return val1
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc_1(x))
        print(x.shape)
        x = F.relu(self.enc_2(x))
        print(x.shape)
        x = F.relu(self.enc_3(x))
        print(x.shape)
        x = F.relu(self.enc_4(x))
        print(x.shape)
        x = self.enc_5(x)
        print(x.shape)
        a = x
        b = x
        x = self.shuffling(a, b)
        # decoding
        x = F.relu(self.dec_1(x))
        print(x.shape)
        x = F.relu(self.dec_2(x))
        print(x.shape)
        x = F.relu(self.dec_3(x))
        print(x.shape)
        x = F.relu(self.dec_4(x))
        print(x.shape)
        reconstruction = torch.sigmoid(self.dec_5(x))
        print(reconstruction.shape)
        return reconstruction, a, b


def plot_pointCloud(pc):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,1,0), (0,0,-1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=-pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            opacity=0.8
        )
    )])
    fig.show()

name1 = "mesh_" + '0051' + '.obj'
ob1 = pywavefront.Wavefront('/content/gdrive/My Drive/CSC420/Project/D_bouncing/meshes/' + name1, strict=True)
plot_pointCloud(np.array(ob1.vertices))