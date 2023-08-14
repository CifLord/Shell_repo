from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import os

def train_fn(data_loader, model, optimizer, device,epoch, optimize_after=8):
    data_loader.sampler.set_epoch(epoch)
    model.train()
    total_loss = 0.0
    iteration = 0
    model=DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)

    for images in tqdm(data_loader):
        images = images.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        targets = images.y/images.natoms   
        loss, acc = get_loss(predictions, targets) 
        loss.backward()

        # Accumulate gradients for a specified number of iterations
        iteration += 1
        if iteration % optimize_after == 0:
            optimizer.step()
            iteration = 0
            total_loss += loss.item()

    return total_loss / (len(data_loader) // optimize_after)

def eval_fn(data_loader,model,device,epoch):
    data_loader.sampler.set_epoch(epoch)
    model.eval()
    total_loss=0.0
    total_acc=0    
    with torch.no_grad():
        for images in tqdm(data_loader):            
            images=images.to(device) 
            predictions = model(images)
            targets = images.y/images.natoms  
            loss, acc = get_loss(predictions, targets)
            total_loss+=loss.item()
            total_acc+=acc.item()

    return total_loss/len(data_loader),total_acc/len(data_loader)



class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, warmup_epochs:int, decay_epochs:int,snapshot_path:str):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gpu_id=int(os.environ["LOCAL_RANK"])
        #self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        
        if os.path.exists(snapshot_path):
            print('Loading snapshot')
            self._load_snapshot(snapshot_path)

        #torch.cuda.set_device(device)
        self.model = model.to(self.gpu_id)
        self.epochs_run=0
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)
        self.decay_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_epochs, gamma=0.9)
        self.best_valid_loss = np.Inf
        self.model=DDP(self.model,device_ids=[self.gpu_id],find_unused_parameters=True)
        
        if self.gpu_id==0:
            wandb.init(project='shell-transformer')
            wandb.watch(self.model)

    def train(self, num_epochs):
        for epoch in range(self.epochs_run,num_epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss, acc = self.evaluate(epoch)

            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.decay_scheduler.step()

            if self.gpu_id==0 and valid_loss < self.best_valid_loss:
                #ckp=self.model.module.state_dict()
                self._save_snapshot(epoch)
                print('saved-model')
                self.best_valid_loss = valid_loss

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.gpu_id==0:
                wandb.log({'epoch': epoch, 'Train_loss': train_loss, 'Valid_loss': valid_loss, 'Valid acc': acc,
                       'lr': current_lr})
                print(f'epoch:{epoch+1} Train_loss:{train_loss} Valid_loss:{valid_loss} Valid acc:{acc} lr:{current_lr}')

    def train_epoch(self, epoch, optimize_after=8):
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        iteration = 0

        for images in tqdm(self.train_loader):
            images = images.to(self.gpu_id)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            targets = images.y 
            num_atoms=images.natoms
            loss, acc = self.get_loss(predictions, targets,num_atoms=num_atoms)
            loss.backward()

            # Accumulate gradients for a specified number of iterations
            iteration += 1
            if iteration % optimize_after == 0:
                self.optimizer.step()
                iteration = 0
                total_loss += loss.item()

        return total_loss / (len(self.train_loader) // optimize_after)

    def evaluate(self, epoch):
        self.val_loader.sampler.set_epoch(epoch)
        self.model.eval()
        total_loss = 0.0
        total_acc = 0

        with torch.no_grad():
            for images in tqdm(self.val_loader):
                images = images.to(self.gpu_id)
                predictions = self.model(images)
                targets = images.y 
                num_atoms=images.natoms
                loss, acc = self.get_loss(predictions, targets,num_atoms=num_atoms)
                total_loss += loss.item()
                total_acc += acc.item()

        return total_loss / len(self.val_loader), total_acc / len(self.val_loader)

    def get_loss(self,predictions, targets,norm=True, num_atoms=1,y_mean=-6.21, y_std=7.26):
        mask_loss = nn.MSELoss()
        mask_acc=nn.L1Loss()
        if norm is True:
            masks = (targets/num_atoms- y_mean) / y_std   
            #print(masks.shape,predictions.shape,targets.shape)         
            pred_back = (predictions*y_std+y_mean)*(num_atoms.view(-1,1))
            loss = mask_loss(predictions, masks.view(-1, 1))   
            accuracy = mask_acc(pred_back ,targets.view(-1, 1))
        else:                      
            loss = mask_loss(predictions.view(-1, 1), targets.view(-1, 1))
            accuracy = mask_acc(predictions.view(-1, 1), targets.view(-1, 1))
        return loss, accuracy
    def _save_snapshot(self,epoch):
        snapshot={}
        model=self.model
        raw_model=model.module if hasattr(model,"module") else model
        snapshot["MODEL_STATE"]=raw_model.state_dict()
        snapshot["EPOCHS_RUN"]=epoch
        torch.save(snapshot,"./params/best_model.pt")
            
    def _load_snapshot(self,snapshot_path):
        snapshot=torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run=snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch{self.epochs_run}")