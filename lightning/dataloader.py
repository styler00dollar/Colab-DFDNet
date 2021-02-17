from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import *

class DFNetDataModule(pl.LightningDataModule):
    def __init__(self, training_path: str = './', train_partpath: str = './', validation_path: str = './', val_partpath: str = './', test_path: str = './', batch_size: int = 5, num_workers: int = 2):
        super().__init__()
        self.training_dir = training_path
        self.validation_dir = validation_path
        self.test_dir = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = 512


        self.train_partpath = train_partpath
        self.val_partpath = val_partpath

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize(size=self.size),
            transforms.CenterCrop(size=self.size),
            #transforms.RandomHorizontalFlip()
            #transforms.ToTensor()
        ])

        self.DFDNetdataset_train = DS(root_dir=self.training_dir, transform=transform, fine_size=self.size, partpath=self.train_partpath)
        self.DFDNetdataset_validation = DS_val(root_dir=self.validation_dir, transform=transform, partpath=self.val_partpath)
        self.DFDNetdataset_test = DS_val(root_dir=self.validation_dir, transform=transform, partpath='/content/landmarks')

    def train_dataloader(self):
        return DataLoader(self.DFDNetdataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.DFDNetdataset_validation, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.DFDNetdataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
