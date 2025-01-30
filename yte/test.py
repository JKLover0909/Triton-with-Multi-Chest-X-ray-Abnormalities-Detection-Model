import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data import XrayDataset
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
import timm

df_val = pd.read_csv("/home/ubuntu/ductq/csvs/val_set.csv")
val_set = XrayDataset(df_val)
val_loader = DataLoader(val_set, batch_size=16, num_workers=8, shuffle=False)


model = timm.create_model("tf_efficientnet_b0", num_classes = 1)

state_dict = torch.load("/home/ubuntu/ductq/results/ckpt/v5/last.ckpt")["state_dict"]
state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.cuda().eval()

pred = torch.tensor([], device="cuda")
gt = torch.tensor([], device="cuda")

for batch in val_loader:
    x, y = batch["image"].cuda().float(), batch["label"].cuda().float()

    with torch.no_grad():
        output = model(x).float()

    predict = torch.sigmoid(output.squeeze(-1))
    
 
    pred = torch.cat((pred, predict),dim=0)
    gt = torch.cat((gt, y), dim=0)

