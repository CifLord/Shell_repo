
import torch.nn as nn
from Models.EGformer import EGformer


class Egformer(nn.Module):

    def __init__(self,**loaded_model_hparams):
        super(Egformer,self).__init__()
        self.arc=EGformer(**loaded_model_hparams)        
        self.mask_loss = nn.MSELoss()
    def norm_label(self,data,y_mean=-7,y_std=6):

        label=data.y/data.natoms
        label=(label-y_mean)/y_std
        return label

    def forward(self,data):
        pred_output = self.arc(data)
        
        masks=self.norm_label(data)
        loss = self.mask_loss(pred_output.view(-1, 1), masks.view(-1, 1))
        acc=sum(abs(pred_output.view(-1, 1)-masks.view(-1, 1)))

        if masks !=None:            
            return loss,acc

        return loss,acc