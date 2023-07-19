from tqdm import tqdm
import torch

def train_fn(data_loader, model, optimizer, device,optimize_after=8):
    model.train()
    total_loss = 0.0
    iteration = 0
    
    for images in tqdm(data_loader):
        images = images.to(device)  
        optimizer.zero_grad()
        loss, acc = model(images)
        loss.backward()
        
        # Accumulate gradients for a specified number of iterations
        iteration += 1
        if iteration % optimize_after == 0:
            optimizer.step()
            iteration = 0
            total_loss += loss.item()
    #scheduler.step() 
    return total_loss / (len(data_loader) // optimize_after)

def eval_fn(data_loader,model,device):
    model.eval()
    total_loss=0.0
    total_acc=0    
    with torch.no_grad():
        for images in tqdm(data_loader):
            #model=model.to(device)
            images=images.to(device)                      
            loss,acc=model(images)
            total_loss+=loss.item()
            total_acc+=acc.item()

    return total_loss/len(data_loader),total_acc/len(data_loader)