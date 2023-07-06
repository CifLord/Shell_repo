import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import pytorch_lightning as pl
from Models.EGformer import EGformer
import torch.nn as nn
import torch.optim as optim
from ocpmodels.datasets import LmdbDataset
from torch.utils.data import random_split
import torch_geometric.loader as geom_loader
import torch_geometric.data as data
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.init as init
import torch.nn as nn
import wandb
wandb.init()
wandb_logger = WandbLogger()
CHECKPOINT_PATH="./checkpoints"

class GeoTransformer_Traniner(pl.LightningModule):
    ''''pytorch lightning'''
    def __init__(self,model,y_mean,y_std,optimizer_name,optimizer_hparams,**model_kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model=model        
        # self.optimizer_name=optimizer_name
        # self.optimizer_hparams=optimizer_hparams
        
        self.loss_module=nn.MSELoss()
        self.y_mean=y_mean
        self.y_std=y_std

    def forward(self,data):
        # x,edge_index,batch_idx=data.latent,data.edge_index,data.batch
        
        x=self.model(data)
        preds=x.squeeze()        
        label=data.y_relaxed/data.natoms
        label=(label-self.y_mean)/self.y_std
        label=label.squeeze()
        loss=self.loss_module(label,preds)
        acc=abs(label-preds)

        return loss,acc
    
    def configure_optimizers(self) -> Any:

        if self.hparams.optimizer_name == "Adam":
            optimizer=optim.AdamW(
                self.parameters(),**self.hparams.optimizer_hparams
            )
        scheduler=optim.lr_scheduler.MultiStepLR(
            optimizer,milestones=[5,10],gamma=0.1
        )    
        # return super().configure_optimizers()
        return [optimizer],[scheduler]


    
    def training_step(self,data,batch_idx):
        loss,acc=self.forward(data)
        self.log("train_loss",loss.mean())
        self.log("train_mae",acc.mean())
        return loss
        
    def validation_step(self,data,batch_idx):
        _,acc=self.forward(data)
        self.log("val_mae",acc.mean())

    def test_step(self,data,batch_idx):
        _,acc=self.forward(data)
        self.log("test_mae",acc.mean())

def train_model(model,train_loader,val_loader,save_name=None,**kwargs):
    pl.seed_everything(42)

    if save_name is None:
        # raise TypeError('need a save name')
        save_name='exmodel'
    
    trainer=pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH,save_name),
                       accelerator='cuda',
                       devices=-1,
                       max_epochs=25,
                       callbacks=[ModelCheckpoint(save_weights_only=True,mode="min",monitor="val_mae"),
                                  LearningRateMonitor("epoch")],
                       enable_progress_bar=True,
                       logger=wandb_logger)
    
    trainer.logger._log_graph=True
    trainer.logger._default_hp_metric=None
    pretrained_filename=os.path.join(CHECKPOINT_PATH,save_name+".ckpt")
    if os.path.isfile(pretrained_filename):
        model=GeoTransformer_Traniner.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        trainer.fit(model,train_loader,val_loader)
        model=GeoTransformer_Traniner.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    val_result=trainer.test(model,val_loader,verbose=False)
    # test_result=trainer.test(model,test_loader,verbose=False)
    # result={"test":test_result[0]["test_acc"],"val":val_result[0]["test_acc"]}

    return model,val_result