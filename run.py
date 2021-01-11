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
from models.model_resnet import MultiScaleDiscriminator
from data.custom_dataset import AlignedDataset
from torch.utils.data import DataLoader
import torchvision

from util.Loss import hinge_loss, hinge_loss_G
from pytorch_msssim import MS_SSIM

from torch.utils.tensorboard import SummaryWriter
from trains import Task
from torchvision.utils import make_grid
from tqdm import tqdm

from util.losses import LossNetworkVgg19


def tensor2im(input_image, norm=1, imtype=np.uint8):
    image_numpy = input_image.data.cpu().float().clamp_(-1,1).numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def train(device, dataset, trainloader, netG, netD, writer, logger):
    optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    criterionG = torch.nn.MSELoss()
    criterionD = torch.nn.BCELoss()
#     criterionD = torch.nn.MSELoss()
#     criterionD = hinge_loss()
    hinge_G = hinge_loss_G()
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
    
    weights_layer_perceptual = [0.5, 1., 2., 4.]
    vgg19_model = torchvision.models.vgg19(pretrained=True)
    vgg19_model.to(device)
    
    perceptual_loss_vgg19 = LossNetworkVgg19(vgg19_model)
    perceptual_loss_vgg19.to(device)
    perceptual_loss_vgg19.eval()
    del vgg19_model

    num_epochs = 15

    for epoch in range(num_epochs):
        mean_loss_G = 0.0
        mean_loss_D = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):

            data_a, data_c = data['A'], data['C']
            data_a = data_a.to(device)
            data_c = data_c.to(device)
            data_part_locations = data['part_locations']

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()

            # Format batch
            real_batch  = {}
            real_batch['1'] = data_c
            real_batch['2'] = F.interpolate(data['C'], (256, 256)).to(device)
            real_batch['4'] = F.interpolate(data['C'], (128, 128)).to(device)
            real_batch['8'] = F.interpolate(data['C'], (64, 64)).to(device)
            label = torch.full((1,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_batch)

            # Calculate loss on all-real batch
            errD_real_1 = criterionD(output['prediction_1'].view(-1), label)
            errD_real_2 = criterionD(output['prediction_2'].view(-1), label)
            errD_real_4 = criterionD(output['prediction_4'].view(-1), label)
            errD_real_8 = criterionD(output['prediction_8'].view(-1), label)
            

            errD_real = errD_real_1 + errD_real_2 + errD_real_4 + errD_real_8


            # Calculate gradients for D in backward pass
            errD_real.backward()


            ## Train with all-fake batch

            fake = netG(data_a, part_locations=data_part_locations)

            # Format batch
            fake_batch  = {}
            fake_batch['1'] = fake.detach()
            fake_batch['2'] = F.interpolate(fake, (256, 256)).to(device)
            fake_batch['4'] = F.interpolate(fake, (128, 128)).to(device)
            fake_batch['8'] = F.interpolate(fake, (64, 64)).to(device)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output_fake = netD(fake_batch)

            # Calculate D's loss on the all-fake batch
            errD_fake_1 = criterionD(output_fake['prediction_1'].view(-1), label)
            errD_fake_2 = criterionD(output_fake['prediction_2'].view(-1), label)
            errD_fake_4 = criterionD(output_fake['prediction_4'].view(-1), label)
            errD_fake_8 = criterionD(output_fake['prediction_8'].view(-1), label)

            errD_fake = errD_fake_1 + errD_fake_2 + errD_fake_4 + errD_fake_8

            # Calculate the gradients for this batch
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            
#             errD.backward()
            
            # Update D
            optimizerD.step()


            ###########################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            fake = netG(data_a, part_locations=data_part_locations)
            mse_loss = criterionG(fake, data_c)
            
            #             VGG_LOSS
            L_p_vgg19 = 0
            perceptual_dst_pred = perceptual_loss_vgg19(fake)
            with torch.no_grad():
                perceptual_dst_img = perceptual_loss_vgg19(data_c)                    
            for k in range(4):
                L_p_vgg19 += weights_layer_perceptual[k] * torch.nn.MSELoss()(perceptual_dst_pred[k],perceptual_dst_img[k])

            L_p = L_p_vgg19
            
            ms_ssim_loss = 1 - ms_ssim_module(fake, data_c)

            # Format batch
            fake_batch  = {}
            fake_batch['1'] = fake
            fake_batch['2'] = F.interpolate(fake, (256, 256)).to(device)
            fake_batch['4'] = F.interpolate(fake, (128, 128)).to(device)
            fake_batch['8'] = F.interpolate(fake, (64, 64)).to(device)
            label.fill_(fake_label)

            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_batch)

            # Calculate G's loss based on this output
#             err_hinge_G = hinge_G(output['prediction_1']) + hinge_G(output['prediction_2']) + hinge_G(output['prediction_4']) + hinge_G(output['prediction_8'])
#             adversarial = 4 * output_fake['prediction_1'] + 2 * output_fake['prediction_2'] + output_fake['prediction_4'] + output_fake['prediction_8']
            # Adversarial loss
            adversarial = 4. - (output['prediction_1'].view(-1) + output['prediction_2'].view(-1) + output['prediction_4'].view(-1) + output['prediction_8'].view(-1))
            errG = mse_loss + L_p + adversarial + 100*ms_ssim_loss

            # Calculate gradients for G
            errG.backward()

            # Update G
            optimizerG.step()
            
            writer.add_scalar('perceptual_loss', L_p.item(), epoch * len(trainloader) + i)
            writer.add_scalar('mse_loss', mse_loss.item(), epoch * len(trainloader) + i)
            writer.add_scalar('adversarial', adversarial.item(), epoch * len(trainloader) + i)
            writer.add_scalar('ms-ssim', 100*ms_ssim_loss.item(), epoch * len(trainloader) + i)
            writer.add_scalar('loss_G', errG, epoch * len(trainloader) + i)
            writer.add_scalar('loss_D', errD, epoch * len(trainloader) + i)
            
            
            # Output training stats
            if i % 100 == 0 and i != 0:                
                torch.save(netG.state_dict(), f'weights/netG_30k_epoch{epoch}_exp2_4.pth')
                torch.save(netD.state_dict(), f'weights/netD_30k_epoch{epoch}_exp2_4.pth')
                
                images = [data_a[0].cpu(), fake[0].cpu(), data_c[0].cpu()]
                report_img = make_grid(images)
                
                logger.report_image('image', f'epoch_{epoch}, iter_{i}', iteration=epoch * len(trainloader) + i, image=tensor2im(report_img))
                logger.flush()

def main():
    
    # tensorboard
    writer = SummaryWriter()

    # trains parameters dict
    parameters_dict = {
        'optimizerG': 'Adam (0.5, 0.99)',
        'optimizerD': 'Adam (0.5, 0.99)',
        'learning_rate': '2e-4',
        'dataset': 'celeba (30k)',
        'resolution': '512',
    }
    
    # init trains
    task = Task.init(project_name='face_enhancement', task_name='Exp. 2.5, res 512, celeba 30k, bce + ms_ssim')
    logger = task.get_logger()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    dataset = AlignedDataset('dataset_celeba/images', fine_size=512)
    trainloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)
    netG = networks.define_G('UNetDictFace', ['cuda:0'])
    netD = MultiScaleDiscriminator(scales=(1, 2, 4, 8))
    netD.to('cuda:0')         
    
#     netG.load_state_dict(torch.load('weights/netG_30k_epoch0_exp2_1.pth'))
#     netD.load_state_dict(torch.load('weights/netD_30k_epoch0_exp2_1.pth'))
    
    cfg_str = str(netG) + str('\n\n Discriminator:\n\n') + str(netD)
    Task.current_task().set_model_config(cfg_str)
    
    # connect the dictionary to TRAINS Task
    parameters_dict = Task.current_task().connect(parameters_dict)
    
    train(device, dataset, trainloader, netG, netD, writer, logger)
    
    
    
if __name__ == '__main__':
    main()
