import torch.nn as nn
from .residual_layer_norm import ResidualLayerNorm
from .mha import MultiHeadAttention
from .pwffn import PWFFN

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()

        self.norm1=ResidualLayerNorm(d_model,dropout)
        self.norm2=ResidualLayerNorm(d_model,dropout)

        self.mha=MultiHeadAttention(d_model,num_heads,dropout)

        self.ff=PWFFN(d_model,d_ff,dropout)

    def forward(self,x,mask):

        mha,encoder_attention_weights=self.mha(x,x,x,mask=mask)

        norm1=self.norm1(mha,x)
        ff=self.ff(norm1)
        norm2=self.norm2(ff,norm1)

        return norm2,encoder_attention_weights