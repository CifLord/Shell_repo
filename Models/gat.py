import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATlayer(GATConv):

    def __init__(self,c_in,c_out,num_layers=8, num_heads=8,) -> None:
        '''Inputs:
                c_in: input feature dimension
                c_out: output feature dimension
                num_heads: the output features are equally split up over the heads if concat_heads=True
                concat_heads: the output of the different heads is concatenated instaed of averaged
                alpha:negative slpoe of the leakyReLU activation
        '''
        super(GATlayer,self).__init__(c_in,c_out)        
        self.conv1=GATConv(c_in,c_out,num_layers=num_layers,num_heads=num_heads,dropout=0.5)        

    def forward(self,h,edge_index):
        ''' h:(nAtoms, emb_size_atom)
            m:(nEdges, emb_size_edge)'''            
        x=self.conv1(h,edge_index) 
               
        return F.softmax(x,dim=-1)