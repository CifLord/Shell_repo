import torch.nn as nn
from Models.EGformer2 import EGformer
import os
import yaml
import torch

def config_model(from_scrach=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    file_path = os.path.join(script_dir, '..', 'params', 'model_hparams.yml')
    with open(file_path, 'r') as file:
        loaded_model_hparams = yaml.load(file, Loader=yaml.FullLoader)['model']

    # Create the model using the loaded hyperparameters
    model = EGformer(**loaded_model_hparams)
    if from_scrach==False:
        checkpoint_path=os.path.join(script_dir, '..', 'params', 'pretrained_model.pt')
        pretrained_state_dict = torch.load(checkpoint_path)["MODEL_STATE"]
        model.load_state_dict(pretrained_state_dict)
        
        return model
    
    else:
        checkpoint_path=os.path.join(script_dir, '..', 'params', 'gemnet_oc_base_s2ef_all_md.pt')
        pretrained_state_dict = torch.load(checkpoint_path)['state_dict']
        new_model_state_dict = model.state_dict()
        filtered_pretrained_state_dict = {k.strip('module.module.'): v for k, v in pretrained_state_dict.items() if k.strip('module.module.') in new_model_state_dict}
        new_model_state_dict.update(filtered_pretrained_state_dict)
        model.load_state_dict(new_model_state_dict)
        for param_name, param in model.named_parameters():
            if param_name in filtered_pretrained_state_dict.keys():                
                param.requires_grad = False                
        return model

# model=GemNetOC(num_atoms=0, bond_feat_dim=0, num_targets=1, num_spherical=7, num_radial=128,
#                num_blocks=4, emb_size_atom=256, emb_size_edge=512, emb_size_trip_in=64, emb_size_trip_out=64, 
#                emb_size_quad_in=32, emb_size_quad_out=32, emb_size_aint_in=64, emb_size_aint_out=64, emb_size_rbf=16, 
#                emb_size_cbf=16, emb_size_sbf=32, num_before_skip=2, num_after_skip=2, num_concat=1, num_atom=3, num_output_afteratom=3) 

# When we want to access some customized attributes of the DDP wrapped model, we must reference model.module. 
# That is to say, our model instance is saved as a module attribute of the DDP model. 
# If we assign some attributes xxx other than built-in properties or functions, we must access them by model.module.xxx.
# When we save the DDP model, our state_dict would add a module prefix to all parameters.
# Consequently, if we want to load a DDP saved model to a non-DDP model, we have to manually strip the extra prefix. I provide my code below:
# in case we load a DDP model checkpoint to a non-DDP model

# model_dict = OrderedDict()
# pattern = re.compile('module.')
# for k,v in state_dict.items():
#     if re.search("module", k):
#         model_dict[re.sub(pattern, '', k)] = v
#     else:
#         model_dict = state_dict
# model.load_state_dict(model_dict)


# History file, may useful in the later.
# class Egformer(nn.Module):

#     def __init__(self,**loaded_model_hparams):
#         super(Egformer,self).__init__()
#         self.arc=EGformer(**loaded_model_hparams)        
#         self.mask_loss = nn.MSELoss()
#     def norm_label(self,data,y_mean=-7,y_std=6):

#         label=data.y/data.natoms
#         label=(label-y_mean)/y_std
#         return label

#     def forward(self,data):
#         pred_output = self.arc(data)        
#         masks=self.norm_label(data)
#         loss = self.mask_loss(pred_output.view(-1, 1), masks.view(-1, 1))
#         acc=sum(abs(pred_output.view(-1, 1)-masks.view(-1, 1)))

#         if masks !=None:            
#             return loss,acc
#         else:
#             return pred_output
