# testing the model
import pytorch_lightning as pl

dm = DS_green_from_mask('/content/test')
model = CustomTrainClass()
# GPU
trainer = pl.Trainer(gpus=1, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
# GPU with AMP (amp_level='O1' = mixed precision)
#trainer = pl.Trainer(gpus=1, precision=16, amp_level='O1', max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
# TPU
#trainer = pl.Trainer(tpu_cores=8, max_epochs=10, progress_bar_refresh_rate=20, default_root_dir='/content/', callbacks=[CheckpointEveryNSteps(save_step_frequency=1000, save_path='/content/')])
trainer.test(model, dm, ckpt_path='/content/Checkpoint_2_2250.ckpt')
