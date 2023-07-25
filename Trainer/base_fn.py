from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

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
    def __init__(self, model, train_loader, val_loader, device, learning_rate, warmup_epochs, decay_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs

        torch.cuda.set_device(device)
        self.model = self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)
        self.decay_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_epochs, gamma=0.1)
        self.best_valid_loss = np.Inf

        wandb.init(project='shell-transformer')
        wandb.watch(self.model)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss, acc = self.evaluate(epoch)

            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.decay_scheduler.step()

            if valid_loss < self.best_valid_loss:
                torch.save(self.model.state_dict(), 'best_model_n.pt')
                print('saved-model')
                self.best_valid_loss = valid_loss

            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({'epoch': epoch, 'Train_loss': train_loss, 'Valid_loss': valid_loss, 'Valid acc': acc,
                       'lr': current_lr})
            print(f'epoch:{epoch+1} Train_loss:{train_loss} Valid_loss:{valid_loss} Valid acc:{acc} lr:{current_lr}')

    def train_epoch(self, epoch, optimize_after=8):
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        iteration = 0

        for images in tqdm(self.train_loader):
            images = images.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            targets = images.y / images.natoms
            loss, acc = self.get_loss(predictions, targets)
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
                images = images.to(self.device)
                predictions = self.model(images)
                targets = images.y / images.natoms
                loss, acc = self.get_loss(predictions, targets)
                total_loss += loss.item()
                total_acc += acc.item()

        return total_loss / len(self.val_loader), total_acc / len(self.val_loader)

    def get_loss(self,predictions, targets, y_mean=-7, y_std=6):
        masks = (targets- y_mean) / y_std
        mask_loss = nn.MSELoss()
        mask_acc=nn.L1Loss()
        loss = mask_loss(predictions.view(-1, 1), masks.view(-1, 1))
        accuracy = mask_acc(predictions.view(-1, 1) , masks.view(-1, 1))
        return loss, accuracy