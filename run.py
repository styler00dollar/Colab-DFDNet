import os
import numpy as np
import cv2

import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
from skimage import transform as trans
from skimage import io
import sys
sys.path.append('FaceLandmarkDetection')
import face_alignment
import dlib
from util.visualizer import save_crop
from util import util
import matplotlib.pyplot as plt

from models import *
from model_resnet import MultiScaleDiscriminator
from custom_dataset import AlignedDataset
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim


def tensor2im(input_image, norm=1, imtype=np.uint8):
    image_numpy = input_image.data.cpu().float().clamp_(-1,1).numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return cv2.cvtColor(image_numpy.astype(imtype), cv2.COLOR_BGR2RGB)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

dataset = AlignedDataset('dataset/ffhq', fine_size=512)
trainloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=os.cpu_count())

netG = networks.define_G('UNetDictFace', ['cuda:0'])
# netD = MultiScaleDiscriminator(scales=(1, 2, 3, 4))
# netD.to('cuda:0')         

criterion = torch.nn.L1Loss()
# optimizer = optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        data_a, data_c = data['A'], data['C']
        data_a = data_a.to(device)
        data_c = data_c.to(device)
        data_part_locations = data['part_locations']
        optimizerG.zero_grad()
        outputs = netG(data_a, part_locations=data_part_locations)
        loss = criterion(outputs, data_c)
        loss.backward()
        optimizerG.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 0:
          print('Epoch %d, iteration %5d, Loss: %.6f' % (epoch + 1, i, running_loss / 100))
          running_loss = 0