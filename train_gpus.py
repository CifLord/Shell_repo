import os
import wandb
import torch
from ocpmodels.datasets import LmdbDataset
from torch.utils.data import random_split
from typing import Any
import torch.nn.init as init
from Trainer.base_fn import Trainer
from Models.EGformer import EGformer
import torch.distributed as dist
import torch.multiprocessing as mp
 # Initialize the distributed environment
from Loader.Dataloader import setup, DistributedDataLoader
from Trainer.instant_model import config_model

warmup_epochs=5
decay_epochs=15   
y_mean=-7
y_std=6
num_epochs=20
batch_size = 4
learning_rate=0.001
#DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset=LmdbDataset({"src":"/shareddata/ocp/ocp22/oc22_trajectories/trajectories/Transformer_clean_valid/"})

def main(snapshot_path:str ="snapshot.pt"):
    
    setup()
    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length
    # Split the dataset into train and validation
    train_dataset, val_dataset =random_split(dataset, [train_length, val_length])
    train_loader = DistributedDataLoader(train_dataset, batch_size=batch_size,drop_last=True)
    val_loader =DistributedDataLoader(val_dataset, batch_size=batch_size,drop_last=True)
    
    # Create the model using the loaded hyperparameters
    # torch.cuda.set_device(rank)   
    model=config_model()    
    
    trainer = Trainer(model, train_loader, val_loader, learning_rate=learning_rate,
                      warmup_epochs=warmup_epochs, decay_epochs=decay_epochs,snapshot_path=snapshot_path)
    trainer.train(num_epochs)
    dist.destroy_process_group()
    
if __name__ == '__main__':
    world_size=4
    main()
    

#model=model.to(rank)
#model=DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)
#criterion=nn.MSELoss()
#optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
#warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: epoch/warmup_epochs)
#decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epochs,gamma=0.1)
# torch 2.0 version
# warmup_lamdb=lambda epoch: epoch/warmup_epochs
# decay_lamdb=lambda epoch: 0.95**epoch
# scheduler=LambdaLR(optimizer,lr_lambda=[warmup_lamdb,decay_lamdb])
#wandb.init(project='shell-transformer')
#wandb.watch(model)    
# best_valid_loss=np.Inf
# for epoch in range(num_epochs):
    
#     train_loss=train_fn(train_loader,model,optimizer,device=rank,epoch=epoch)
#     valid_loss,acc=eval_fn(val_loader,model,device=rank,epoch=epoch)
#     if epoch<warmup_epochs:
#         warmup_scheduler.step()
#     else:
#         decay_scheduler.step()    
#     if valid_loss< best_valid_loss:
#         torch.save(model.module.state_dict(),'best_model_n.pt')
#         print('saved-model')
#         best_valid_loss=valid_loss
#     current_lr=optimizer.param_groups[0]['lr']
#     wandb.log({'epoch':epoch,'Train_loss':train_loss, 'Valid_loss':valid_loss, 'Valid acc':acc,'lr':current_lr})
#     print(f'epoch:{epoch+1} Train_loss:{train_loss} Valid_loss:{valid_loss} Valid acc:{acc} lr:{current_lr}')
# dist.destroy_process_group()