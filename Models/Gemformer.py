import torch 
import torch.nn as nn
from torch_scatter import scatter

class Gemformer(nn.Module):
    '''This is a Graph Transformer neural network, 
    the input is the Gemnet-oc latent space with the best pre-trained params
    
    Parameters
    num_heads:
        Number of heads
    '''

    def __init__(self,num_heads,emb_size_in,emb_size_trans,out_layer1=32,out_layer2=1):
        super(Gemformer, self).__init__()
        self.num_heads=num_heads
        
        self.out_layer1=out_layer1
        self.out_layer2=out_layer2
        self.dense=nn.Sequential(nn.Linear(emb_size_trans,out_layer1),
                                 nn.SiLU(),
                                 nn.Linear(out_layer1,out_layer2)                                 
                                 )

        self.lin_query_MHA=nn.Linear(emb_size_in,emb_size_trans)
        self.lin_key_MHA=nn.Linear(emb_size_in,emb_size_trans)
        self.lin_value_MHA=nn.Linear(emb_size_in,emb_size_trans)

        self.softmax=nn.Softmax(dim=1)
        #--------------------------------------------------need update------------------------------------------------------
        num_layers=3

        self.MHA=nn.MultiheadAttention(embed_dim=emb_size_trans,
                                       num_heads=num_heads,
                                       bias=True,
                                       dropout=0.0,
                                       )
        self.encoder_layers=nn.TransformerEncoderLayer(embed_dim=emb_size_trans,num_heads=num_heads,dropout=0.0)
        self.transformer_encoder=nn.TransformerEncoder(self.encoder_layers,num_layers)
        self.layer_norm = nn.LayerNorm(emb_size_trans)
        
    def check_shape(self,va):
        '''check the varaible shape,mean and std,
           only for develop the model'''
        print(f'The {va.__class__.__name__} shape is:',va.shape)
        print(f'The {va.__class__.__name__} mean and std of is:',va.mean(),va.std())
        

    def forward(self,data):

        E_all= data.latent 
        batch = data.batch
        # print(batch)
        q=self.lin_query_MHA(E_all)
        k=self.lin_key_MHA(E_all)
        v=self.lin_value_MHA(E_all)

        nMolecules = torch.max(batch) + 1
        E_t,w=self.MHA(q,k,v)
        # self.check_shape(E_t)
        E_t=torch.sum(E_t,dim=0)
        # self.check_shape(E_t)
        E_t = self.layer_norm(E_t)
        # self.check_shape(E_t)
        # E_t = E_t.permute(1, 0, 2)
        E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        # self.check_shape(E_t)
        
        E_t=self.dense(E_t)
        # self.check_shape(E_t)
        
        return E_t