import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, EarlyStopping
from data import *
from utils import train_transform
# import timm
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from runner import Runner
import pandas as pd


# data
df_train = pd.read_csv("/home/ubuntu/ductq/csvs/train_set.csv")
df_val = pd.read_csv("/home/ubuntu/ductq/csvs/val_set.csv")
# print(df_train)

train_tf = train_transform()
train_set = XrayDataset(df_train, train_transform=train_tf)
val_set = XrayDataset(df_val)

train_loader = DataLoader(train_set, batch_size=16, num_workers=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, num_workers=8, shuffle=False)

# print(val_set[0]['image'].shape)
# print(train_set[0]['image'].shape)
# print(len(train_set))
# print(len(val_set))


# configs
configs = {
    "model_name": "convnextv2_atto",
    "max_epochs": 20,
    "learning_rate": 0.0001,
    "loss": "focal"
}


# callbacks
callback = [
    ModelSummary(max_depth=2),
    TQDMProgressBar(), 
    LearningRateMonitor(logging_interval='step'), 
    ModelCheckpoint(monitor='val_loss', 
                    dirpath='/home/ubuntu/ductq/results/ckpt/v6', 
                    filename="{epoch}-{val_loss:.2f}",
                    mode="min",
                    save_last = True),]
    # EarlyStopping(monitor='val_loss', patience=1, mode='min')]



#logger 
logger = TensorBoardLogger("/home/ubuntu/ductq/results/logger", name="v6", version=None)


# trainer
trainer = Trainer(accelerator="gpu",
                logger=logger,
                # devices = [0,1],
                max_epochs=configs["max_epochs"],
                callbacks=callback,
                enable_progress_bar=True,
                enable_checkpointing=True,
                num_sanity_val_steps=2,
                precision=32, #half precision
                check_val_every_n_epoch=1
)


#runner 
runner = Runner(configs)


# train
trainer.fit(runner,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
