from deep_learning import *
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


if __name__ == "__main__":
    # load iterator
    train_img = []
    #img_set = []
    end = 175
    for i in range(0, end):
        #if img_set != []:
        #    train_img.append(np.array(img_set))
        img_set = []
        for j in range(1, 9, 2):
            str_i = str(i)
            if i < 10:
                str_i = "000" + str_i
            elif i < 100:
                str_i = "00" + str_i
            else:
                str_i = "0" + str_i
            name = 'Image' + str(j) + '_' + str_i + '.png'
            im = cv2.cvtColor(cv2.imread('/content/gdrive/My Drive/CSC420/Project/D_bouncing/images/' + name, 1), cv2.COLOR_BGR2RGB)
            im1 = im[200:1000, 200:1400, :]
            train_img.append(im1)
            plt.imshow(im1), plt.show()
            #img_set.append(np.array(im))
            #print(len(img_set)%2==0)
            #if (j-1) % 2 == 1:
            #    train_img.append(np.array(img_set))
            #    img_set = []
    #train_img.append(np.array(img_set))
    print("len(train_img)", len(train_img))
    labels_train = []
    for i in range(0, end):
        str_i = str(i)
        if i < 10:
            str_i = "000" + str_i
        elif i < 100:
            str_i = "00" + str_i
        else:
            str_i = "0" + str_i
        name = "mesh_" + str_i + '.obj'
        ob = pywavefront.Wavefront('/content/gdrive/My Drive/CSC420/Project/D_bouncing/meshes/' + name, strict=True)
        #if i == 0:
        #    plot_pointCloud(np.array(ob.vertices))
        for q in range(4):
            labels_train.append(np.array(ob.vertices))
    print("len(labels_train)", len(labels_train))
    
    end2 = 100
    val_img = []
    for i in range(0, end2):
        #if img_set != []:
        #    val_img.append(np.array(img_set))
        img_set = []
        for j in range(1, 9, 2):
            str_i = str(i)
            if i < 10:
                str_i = "000" + str_i
            elif i < 100:
                str_i = "00" + str_i
            else:
                str_i = "0" + str_i
            name = 'Image' + str(j) + '_' + str_i + '.png'
            im = cv2.cvtColor(cv2.imread('/content/gdrive/My Drive/CSC420/Project/D_bouncing/images/' + name, 1), cv2.COLOR_BGR2RGB)
            im1 = im[200:1000, 200:1400, :]
            val_img.append(im1)
            plt.imshow(im1), plt.show()
            #img_set.append(np.array(im))
            #print(len(img_set)%2==0)
            #if (j-1) % 2 == 1:
            #    val_img.append(np.array(img_set))
            #    img_set = []
    #val_img.append(np.array(img_set))
    print("len(val_img)", len(val_img))
    labels_val = []
    for i in range(0, end2):
        str_i = str(i)
        if i < 10:
            str_i = "000" + str_i
        elif i < 100:
            str_i = "00" + str_i
        else:
            str_i = "0" + str_i
        name = "mesh_" + str_i + '.obj'
        ob = pywavefront.Wavefront('/content/gdrive/My Drive/CSC420/Project/D_bouncing/meshes/' + name, strict=True)
        #if i == 0:
        #    plot_pointCloud(np.array(ob.vertices))
        for q in range(4):
            labels_val.append(np.array(ob.vertices))
    print("len(labels_val)", len(labels_val))

    end1 = 100
    test_img = []
    for i in range(0, end1):
        #if img_set != []:
        #    test_img.append(np.array(img_set))
        img_set = []
        for j in range(1, 9, 2):
            str_i = str(i)
            if i < 10:
                str_i = "000" + str_i
            elif i < 100:
                str_i = "00" + str_i
            else:
                str_i = "0" + str_i
            name = 'Image' + str(j) + '_' + str_i + '.png'
            im = cv2.cvtColor(cv2.imread('/content/gdrive/My Drive/CSC420/Project/D_bouncing/images/' + name, 1), cv2.COLOR_BGR2RGB)
            im1 = im[200:1000, 200:1400, :]
            test_img.append(im1)
            plt.imshow(im1), plt.show()
            #img_set.append(np.array(im))
            #print(len(img_set)%2==0)
            #if (j-1) % 2 == 1:
            #    test_img.append(np.array(img_set))
            #    img_set = []
    #test_img.append(np.array(img_set))
    print("len(test_img)", len(test_img))
    labels_test = []
    for i in range(0, end1):
        str_i = str(i)
        if i < 10:
            str_i = "000" + str_i
        elif i < 100:
            str_i = "00" + str_i
        else:
            str_i = "0" + str_i
        name = "mesh_" + str_i + '.obj'
        ob = pywavefront.Wavefront('/content/gdrive/My Drive/CSC420/Project/D_bouncing/meshes/' + name, strict=True)
        #if i == 0:
        #    plot_pointCloud(np.array(ob.vertices))
        for q in range(4):
            labels_test.append(np.array(ob.vertices))
    print("len(labels_test)", len(labels_test))

    batch_size1 = 16
    train_loader = load_data(batch_size1, labels_train, train_img, labels_val, val_img, labels_test, test_img)
    training(batch_size=batch_size1, lr=1e-3, epochs=1, enc_weights='', dec_weights='', cuda1=False, train_loader=train_loader)