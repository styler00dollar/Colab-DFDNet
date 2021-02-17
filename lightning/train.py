# Training
import pytorch_lightning as pl

dm = DFNetDataModule(training_path = '/content/DFDNet/ffhq/', train_partpath = '/content/DFDNet/landmarks', validation_path = '/content/validation/', val_partpath='/content/landmarks', batch_size=1)
model = CustomTrainClass()
#model = model.load_from_checkpoint('/content/Checkpoint_0_450.ckpt') # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset

# GPU
trainer = pl.Trainer(logger=None, gpus=1, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])
# 2+ GPUS (locally, not inside Google Colab)
# Recommended: Pytorch 1.8+. 1.7.1 seems to have dataloader issues and ddp seems to cause problems in general.
#trainer = pl.Trainer(logger=None, gpus=2, distributed_backend='dp', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=100, save_path='/content/')])
# GPU with AMP (amp_level='O1' = mixed precision)
# currently not working
#trainer = pl.Trainer(logger=None, gpus=1, precision=16, amp_level='O1', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
# TPU
#trainer = pl.Trainer(logger=None, tpu_cores=8, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
trainer.fit(model, dm)
