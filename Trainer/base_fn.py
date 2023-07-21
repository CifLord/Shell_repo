from tqdm import tqdm
import torch
import torch.nn as nn

def get_loss(predictions, targets, y_mean=-7, y_std=6):
    masks = (targets/ targets.natoms - y_mean) / y_std
    mask_loss = nn.MSELoss()
    loss = mask_loss(predictions.view(-1, 1), masks.view(-1, 1))
    accuracy = torch.abs(predictions.view(-1, 1) - masks.view(-1, 1))
    return loss, accuracy

# def train_fn(data_loader, model, optimizer, device,optimize_after=8):
#     model.train()
#     total_loss = 0.0
#     iteration = 0
    
#     for images in tqdm(data_loader):
#         images = images.to(device)  
#         optimizer.zero_grad()
#         loss, acc = model(images)
#         loss.backward()
        
#         # Accumulate gradients for a specified number of iterations
#         iteration += 1
#         if iteration % optimize_after == 0:
#             optimizer.step()
#             iteration = 0
#             total_loss += loss.item()
#     #scheduler.step() 
#     return total_loss / (len(data_loader) // optimize_after)

# def eval_fn(data_loader,model,device):
#     model.eval()
#     total_loss=0.0
#     total_acc=0    
#     with torch.no_grad():
#         for images in tqdm(data_loader):
#             #model=model.to(device)
#             images=images.to(device)                      
#             loss,acc=model(images)
#             total_loss+=loss.item()
#             total_acc+=acc.item()

#     return total_loss/len(data_loader),total_acc/len(data_loader)

def train_fn(data_loader, model, optimizer, device, optimize_after=8):
    data_loader.sampler.set_epoch(epoch)
    model.train()
    total_loss = 0.0
    iteration = 0

    for images in tqdm(data_loader):
        images = images.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        targets = images.y
        loss, acc = get_loss(predictions, targets)
        loss.backward()

        # Accumulate gradients for a specified number of iterations
        iteration += 1
        if iteration % optimize_after == 0:
            optimizer.step()
            iteration = 0
            total_loss += loss.item()

    return total_loss / (len(data_loader) // optimize_after)

def eval_fn(data_loader,model,device):
    data_loader.sampler.set_epoch(epoch)
    model.eval()
    total_loss=0.0
    total_acc=0    
    with torch.no_grad():
        for images in tqdm(data_loader):
            
            images=images.to(device) 
            predictions = model(images)
            targets = images.y
            loss, acc = get_loss(predictions, targets)
            total_loss+=loss.item()
            total_acc+=acc.item()

    return total_loss/len(data_loader),total_acc/len(data_loader)