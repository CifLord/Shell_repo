import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch

import torch.nn as nn
import torch.optim as optim
from ocpmodels.datasets import LmdbDataset
from torch.utils.data import random_split
import torch_geometric.loader as geom_loader
import torch_geometric.data as data
from typing import Any
import yaml
import torch.nn.init as init
import torch.nn as nn
from Trainer.base_fn import train_fn,eval_fn
from Trainer.instant_model import Egformer

with open('params/model_hparams.yml', 'r') as file:
    loaded_model_hparams = yaml.load(file, Loader=yaml.FullLoader)

# Create the model using the loaded hyperparameters
model = Egformer(**loaded_model_hparams)
checkpoint_path='params/gemnet_oc_base_oc20_oc22.pt'
pretrained_state_dict = torch.load(checkpoint_path)['state_dict']
new_model_state_dict = model.arc.state_dict()
filtered_pretrained_state_dict = {k.strip('module.module.'): v for k, v in pretrained_state_dict.items() if k.strip('module.module.') in new_model_state_dict}
new_model_state_dict.update(filtered_pretrained_state_dict)
model.arc.load_state_dict(new_model_state_dict)
for param_name, param in model.named_parameters():
    if param_name.replace('arc.','') in filtered_pretrained_state_dict.keys():        
        param.requires_grad = False
        
        
y_mean=-7
y_std=6
num_epochs=30
batch_size = 2
learning_rate=0.01
CHECKPOINT_PATH="./checkpoints"
DEVICE='cuda'
dataset=LmdbDataset({"src":"/shareddata/ocp/ocp22/oc22_trajectories/trajectories/Transformer_clean_valid/data.0000.lmdb"})
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

train_length = int(0.8 * len(dataset))
val_length = len(dataset) - train_length

# Split the dataset into train and validation
train_dataset, val_dataset =random_split(dataset, [train_length, val_length])
train_loader = geom_loader.DataLoader(train_dataset, batch_size=6,drop_last=True)
val_loader = geom_loader.DataLoader(val_dataset, batch_size=6,drop_last=True)


best_valid_loss=np.Inf
for i in range(num_epochs):
    train_loss=train_fn(train_loader,model,optimizer,device=DEVICE)
    valid_loss,acc=eval_fn(val_loader,model,device=DEVICE)
    print(acc)
    
    if valid_loss< best_valid_loss:
        torch.save(model.state_dict(),'best_model_n.pt')
        print('saved-model')
        best_valid_loss=valid_loss

    print(f'epoch:{i+1} Train_loss:{train_loss} Valid_loss:{valid_loss} Valid acc:{acc}')