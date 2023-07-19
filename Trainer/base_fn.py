from tqdm import tqdm
import torch

def train_fn(data_loader,model,optimizer,device):
    model.train()
    total_loss=0.0
    
    for images in tqdm(data_loader):
        images=images.to(device)  
        model=model.to(device)     
        optimizer.zero_grad()
        loss,acc=model(images)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
    return total_loss/len(data_loader)

def eval_fn(data_loader,model,device):
    model.eval()
    total_loss=0.0
    total_acc=0    
    with torch.no_grad():
        for images in tqdm(data_loader):
            model=model.to(device)
            images=images.to(device)                      
            loss,acc=model(images)
            total_loss+=loss.item()
            total_acc+=acc.item()

    return total_loss/len(data_loader),total_acc/len(data_loader)