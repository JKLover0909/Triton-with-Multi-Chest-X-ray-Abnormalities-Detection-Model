import torch
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import get_loss_function
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
import timm


class Runner(L.LightningModule):
    def __init__(self, config):  
        super(Runner, self).__init__()

        self.config = config

        self.model_name = self.config["model_name"]
        self.loss_name = self.config["loss"]

        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()

        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=1).float().cuda()
        self.loss = get_loss_function(self.loss_name)

        self.max_epochs = self.config["max_epochs"]
        self.lr = self.config["learning_rate"]

        self.save_hyperparameters()


    def training_step(self, batch):
        # training_step defines the train loop.
        x, y = batch["image"].float().cuda(), batch["label"].float().cuda()

        output = self.model(x).float()

        loss = self.loss(output.squeeze(-1), y)

        pred = torch.sigmoid(output)

        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc(pred.squeeze(-1), y))

        return loss

    

    def validation_step(self, batch, batch_idx):
        # this is the val loop
        x, y = batch["image"].float().cuda(), batch["label"].float().cuda()

        with torch.no_grad():
            output = self.model(x).float()

        loss = self.loss(output.squeeze(-1), y)

        pred = torch.sigmoid(output)
        pred = torch.squeeze(pred)

        self.log("val_loss", loss)
        self.log("val_acc_step", self.valid_acc(pred.squeeze(-1), y))
    


    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # gamma = (1e-6/self.lr)**(1/self.max_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}