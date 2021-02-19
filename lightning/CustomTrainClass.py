from vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from vic.perceptual_loss import PerceptualLoss
from metrics import *
from torch.autograd import Variable
from model import UNetDictFace

import pytorch_lightning as pl
from torchvision.utils import save_image

from init import weights_init
from discriminator import *

import os

from tensorboardX import SummaryWriter
logdir='/content/logs/'
writer = SummaryWriter(logdir=logdir)


class CustomTrainClass(pl.LightningModule):
  def __init__(self):
    super().__init__()

    # generator
    self.netG = UNetDictFace(64)
    weights_init(self.netG, 'kaiming')

    # discriminator
    self.netD = context_encoder()

    # VGG
    #self.netD = Discriminator_VGG(size=256, in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN')
    #self.netD = Discriminator_VGG_fea(size=256, in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D',
    #     arch='ESRGAN', spectral_norm=False, self_attention = False, max_pool=False, poolsize = 4)
    #self.netD = Discriminator_VGG_128_SN()
    #self.netD = VGGFeatureExtractor(feature_layer=34,use_bn=False,use_input_norm=True,device=torch.device('cpu'),z_norm=False)

    # PatchGAN
    #self.netD = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
    #    use_sigmoid=False, getIntermFeat=False, patch=True, use_spectral_norm=False)

    # Multiscale
    #self.netD = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
    #             use_sigmoid=False, num_D=3, getIntermFeat=False)

    # ResNet
    #self.netD = Discriminator_ResNet_128(in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA')
    #self.netD = ResNet101FeatureExtractor(use_input_norm=True, device=torch.device('cpu'), z_norm=False)

    # MINC
    #self.netD = MINCNet()

    # Pixel
    #self.netD = PixelDiscriminator(input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d)

    # EfficientNet
    #from efficientnet_pytorch import EfficientNet
    #self.netD = EfficientNet.from_pretrained('efficientnet-b0')

    # ResNeSt
    # ["resnest50", "resnest101", "resnest200", "resnest269"]
    #self.netD = resnest50(pretrained=True)

    weights_init(self.netD, 'kaiming')


    # loss functions
    self.l1 = nn.L1Loss()
    l_hfen_type = L1CosineSim()
    self.HFENLoss = HFENLoss(loss_f=l_hfen_type, kernel='log', kernel_size=15, sigma = 2.5, norm = False)
    self.ElasticLoss = ElasticLoss(a=0.2, reduction='mean')
    self.RelativeL1 = RelativeL1(eps=.01, reduction='mean')
    self.L1CosineSim = L1CosineSim(loss_lambda=5, reduction='mean')
    self.ClipL1 = ClipL1(clip_min=0.0, clip_max=10.0)
    self.FFTloss = FFTloss(loss_f = torch.nn.L1Loss, reduction='mean')
    self.OFLoss = OFLoss()
    self.GPLoss = GPLoss(trace=False, spl_denorm=False)
    self.CPLoss = CPLoss(rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False)
    self.StyleLoss = StyleLoss()
    self.TVLoss = TVLoss(tv_type='tv', p = 1)
    self.PerceptualLoss = PerceptualLoss(model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], model_path=None)
    layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
    self.Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
        distance_type = 'cosine', b=1.0, band_width=0.5,
        use_vgg = True, net = 'vgg19', calc_type = 'regular')

    self.MSELoss = torch.nn.MSELoss()
    self.L1Loss = nn.L1Loss()

    # metrics
    """
    self.psnr_metric = PSNR()
    self.ssim_metric = SSIM()
    self.ae_metric = AE()
    self.mse_metric = MSE()
    """

  def forward(self, ):
    return self.netG()

  def training_step(self, train_batch, batch_idx):

      #return {'A': A, 'C': C, 'path': path, 'part_locations': part_locations}
      # train_batch[0] = A # lr
      # train_batch[1] = C # hr
      # train_batch[3] = part_locations

      data_part_locations = train_batch[2]

      ####################################
      # generator training
      out = self.netG(train_batch[0], part_locations=train_batch[3])

      # range [-1, 1] to [0, 1]
      out = out + 1
      out = out - out.min()
      out = out / (out.max() - out.min())



      ############################
      # loss calculation
      total_loss = 0
      """
      HFENLoss_forward = self.HFENLoss(out, train_batch[1])
      total_loss += HFENLoss_forward
      ElasticLoss_forward = self.ElasticLoss(out, train_batch[1])
      total_loss += ElasticLoss_forward
      RelativeL1_forward = self.RelativeL1(out, train_batch[1])
      total_loss += RelativeL1_forward
      """
      L1CosineSim_forward = 5*self.L1CosineSim(out, train_batch[1])
      total_loss += L1CosineSim_forward
      #self.log('loss/L1CosineSim', L1CosineSim_forward)
      writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, self.trainer.global_step)

      """
      ClipL1_forward = self.ClipL1(out, train_batch[1])
      total_loss += ClipL1_forward
      FFTloss_forward = self.FFTloss(out, train_batch[1])
      total_loss += FFTloss_forward
      OFLoss_forward = self.OFLoss(out)
      total_loss += OFLoss_forward
      GPLoss_forward = self.GPLoss(out, train_batch[1])
      total_loss += GPLoss_forward

      CPLoss_forward = 0.1*self.CPLoss(out, train_batch[1])
      total_loss += CPLoss_forward


      Contextual_Loss_forward = self.Contextual_Loss(out, train_batch[1])
      total_loss += Contextual_Loss_forward
      self.log('loss/contextual', Contextual_Loss_forward)
      """

      #style_forward = 240*self.StyleLoss(out, train_batch[1])
      #total_loss += style_forward
      #self.log('loss/style', style_forward)

      tv_forward = 0.0000005*self.TVLoss(out)
      total_loss += tv_forward
      #self.log('loss/tv', tv_forward)
      writer.add_scalar('loss/tv', tv_forward, self.trainer.global_step)

      perceptual_forward = 2*self.PerceptualLoss(out, train_batch[1])
      total_loss += perceptual_forward
      #self.log('loss/perceptual', perceptual_forward)
      writer.add_scalar('loss/perceptual', perceptual_forward, self.trainer.global_step)





      # train discriminator
      Tensor = torch.cuda.FloatTensor #if cuda else torch.FloatTensor
      valid = Variable(Tensor(out.shape).fill_(1.0), requires_grad=False)
      fake = Variable(Tensor(out.shape).fill_(0.0), requires_grad=False)
      dis_real_loss = self.MSELoss(train_batch[1], valid)
      dis_fake_loss = self.MSELoss(out, fake)

      d_loss = (dis_real_loss + dis_fake_loss) / 2
      #self.log('loss/d_loss', d_loss)
      writer.add_scalar('loss/g_loss', total_loss, self.trainer.global_step)

      return total_loss+d_loss



  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer

  def validation_step(self, train_batch, train_idx):
    # return {'A': A, 'path': path, 'part_locations': part_locations}
    # train_batch[0] = lr
    # train_batch[1] = path
    # train_batch[2] = part_locations
    out = self.netG(train_batch[0], part_locations=train_batch[2])

    # range [-1, 1] to [0, 1]
    out = out + 1
    out = out - out.min()
    out = out / (out.max() - out.min())


    """
    # metrics
    # currently not usable, but are correctly implemented, needs a modified dataloader
    self.log('metrics/PSNR', self.psnr_metric(train_batch[2], out))
    self.log('metrics/SSIM', self.ssim_metric(train_batch[2], out))
    self.log('metrics/MSE', self.mse_metric(train_batch[2], out))
    self.log('metrics/LPIPS', self.PerceptualLoss(out, train_batch[2]))
    """

    validation_output = '/content/validation_output/'

    # train_batch[3] can contain multiple files, depending on the batch_size
    for f in train_batch[1]:
      # data is processed as a batch, to save indididual files, a counter is used
      counter = 0
      if not os.path.exists(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0])):
        os.makedirs(os.path.join(validation_output, os.path.splitext(os.path.basename(f))[0]))

      filename_with_extention = os.path.basename(f)
      filename = os.path.splitext(filename_with_extention)[0]
      save_image(out[counter], os.path.join(validation_output, filename, str(self.trainer.global_step) + '.png'))



  def test_step(self, train_batch, train_idx):
    # train_batch[0] = masked
    # train_batch[1] = mask
    # train_batch[2] = path
    print("testing")
    test_output = '/content/test_output/'
    if not os.path.exists(test_output):
      os.makedirs(test_output)

    out = self(train_batch[0].unsqueeze(0),train_batch[1].unsqueeze(0))
    out = train_batch[0]*(train_batch[1])+out*(1-train_batch[1])

    save_image(out, os.path.join(test_output, os.path.splitext(os.path.basename(train_batch[2]))[0] + '.png'))
