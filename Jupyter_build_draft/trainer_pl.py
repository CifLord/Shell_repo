import yaml
import sys

from Models.EGformer import EGformer
import torch
from Trainer.config import GeoTransformer_Traniner,train_model
from ocpmodels.datasets import LmdbDataset
from torch.utils.data import random_split
import wandb
import torch_geometric.loader as geom_loader
from pytorch_lightning.loggers import WandbLogger




with open('params/model_hparams.yml', 'r') as file:
    loaded_model_hparams = yaml.load(file, Loader=yaml.FullLoader)

# Create the model using the loaded hyperparameters
model = EGformer(**loaded_model_hparams)
checkpoint_path='params/gemnet_oc_base_oc20_oc22.pt'
pretrained_state_dict = torch.load(checkpoint_path)['state_dict']
new_model_state_dict = model.state_dict()
filtered_pretrained_state_dict = {k.strip('module.module.'): v for k, v in pretrained_state_dict.items() if k.strip('module.module.') in new_model_state_dict}
new_model_state_dict.update(filtered_pretrained_state_dict)
model.load_state_dict(new_model_state_dict)

for param_name, param in model.named_parameters():
    if param_name in filtered_pretrained_state_dict.keys():        
        param.requires_grad = False

y_mean=-2
y_std=2
model=GeoTransformer_Traniner(model=model,y_mean=y_mean,y_std=y_std,optimizer_name="Adam",optimizer_hparams={"lr":1e-3,"weight_decay":1e-4})

dataset=LmdbDataset({"src":"Data/eoh.lmdb"})
train_length = int(0.8 * len(dataset))
val_length = len(dataset) - train_length
# Split the dataset into train and validation
train_dataset, val_dataset =random_split(dataset, [train_length, val_length])
train_loader = geom_loader.DataLoader(train_dataset, batch_size=2)
val_loader = geom_loader.DataLoader(val_dataset, batch_size=2)

gemformer_model,gemformer_results=train_model(model,train_loader,val_loader)
wandb.finish()