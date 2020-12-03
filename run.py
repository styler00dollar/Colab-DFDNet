import os
import numpy as np
import cv2

from data import CreateDataLoader
from models import create_model
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


dataset = AlignedDataset('ffhq', fine_size=512)
trainloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=os.cpu_count())

netG = networks.define_G('UNetDictFace', ['cuda:0'])
netD = MultiScaleDiscriminator(scales=(1, 2, 3, 4))
netD.to('cuda:0')


criterion = torch.nn.L1Loss()
optimizer = optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        data_a, data_c = data['A'], data['C']
        optimizer.zero_grad()
        outputs = netG(data_a, part_locations=data['part_locations'])
        loss = criterion(outputs, data_c)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))


print('Success!!!')