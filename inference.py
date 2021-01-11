import os
import numpy as np
import cv2

import math
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import random
from skimage import transform as trans
from skimage import io
import sys
sys.path.append('FaceLandmarkDetection')
import face_alignment
import dlib

from models import *
from custom_dataset import AlignedDataset
from torch.utils.data import DataLoader
import torchvision

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from trains import Task
from torchvision.utils import make_grid

from options.test_options import TestOptions
from data.image_folder import make_dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

dataset = AlignedDataset('test_dataset/images', fine_size=512)
trainloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=8)
netG = networks.define_G('UNetDictFace', ['cuda:0'])

# # tensorboard
# writer = SummaryWriter()

# # trains parameters dict
# parameters_dict = {
#     'test': 'test',
#     }

# # init trains
# task = Task.init(project_name='face_enhancement', task_name='test')
# logger = task.get_logger()

# cfg_str = str(netG) + str('\n\n Discriminator:\n\n') + str(netD)
# Task.current_task().set_model_config(cfg_str)

# # connect the dictionary to TRAINS Task
# parameters_dict = Task.current_task().connect(parameters_dict)

netG.load_state_dict(torch.load('weights/netG_30k_epoch4_exp2_3.pth'))

for i, data in enumerate(tqdm(trainloader), 0):
    data_a, data_c = data['A'], data['C']
    data_a = data_a.to(device)
    data_c = data_c.to(device)
    data_part_locations = data['part_locations']

    out = netG(data_a, part_locations=data_part_locations)
    images = [data_a[0].cpu(), out[0].cpu()]
    report_img = make_grid(images)
#     report_img = report_img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    result = 255 * (report_img.permute(1, 2, 0).cpu().detach().numpy() + 1) / 2
    cv2.imwrite(f'test_dataset/images2/{i}.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


