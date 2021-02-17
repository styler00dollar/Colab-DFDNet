# check data
import pytorch_lightning as pl
from CustomTrainClass_check import CustomTrainClass
from dataloader import DFNetDataModule

dm = DFNetDataModule(training_path = '/content/DFDNet/ffhq/', train_partpath = '/content/DFDNet/landmarks', validation_path = '/content/validation/', val_partpath='/content/landmarks', batch_size=1)
model = CustomTrainClass()
trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20, default_root_dir='/content/')
trainer.fit(model, dm)
