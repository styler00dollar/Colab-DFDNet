from torch.autograd import Variable

import pytorch_lightning as pl
from torchvision.utils import save_image

class CustomTrainClass(pl.LightningModule):
  def __init__(self):
    super().__init__()

    # generator
    self.netG = UNetDictFace(64)
    weights_init(self.netG, 'kaiming')

    #self.MSELoss = torch.nn.MSELoss()


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
      delete_status = self.netG(train_batch[0], part_locations=train_batch[3])
      if delete_status == 1:
        print(train_batch[2])
      #return 123



  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer

  def validation_step(self, train_batch, train_idx):
    print("val_skip")



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
