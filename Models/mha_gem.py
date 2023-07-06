import torch.nn as nn
import math as m
import torch.nn.functional as F
import torch

class GemMultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super().__init__()
        self.d=d_model//num_heads
        self.d_model=d_model
        self.num_heads=num_heads
        self.dropout=nn.Dropout(dropout)
        self.linear_Q=nn.ModuleList([nn.Linear(d_model,self.d)
                                     for _ in range(num_heads)])
        self.linear_K=nn.ModuleList([nn.Linear(d_model,self.d)
                                     for _ in range(num_heads)])        
        self.linear_V=nn.ModuleList([nn.Linear(d_model,self.d)
                                     for _ in range(num_heads)])
        self.mha_linear=nn.Linear(d_model,d_model)
    
    def scaled_dot_product_attention(self, Q,K,V,mask=None):
        #shape(Q) = [B x Seq x D/num_heads]

        Q_K_matmul=torch.matmul(Q,K.permute(0,2,1))
        scores=Q_K_matmul/m.sqrt(self.d)
        if mask is not None:
            scores=scores.masked_fill(mask==0,1e-9)

        attention_weights=F.softmax(scores,dim=-1)
        output=torch.matmul(attention_weights,V)

        return output, attention_weights
    
    def forward(self,pre_q,pre_k,pre_v,mask=None):
        #shape(x)=[B x seq x D]
        Q=[Linear_Q(pre_q) for Linear_Q in self.linear_Q ]
        K=[Linear_K(pre_k) for Linear_K in self.linear_K ]
        V=[Linear_V(pre_v) for Linear_V in self.linear_V ]
        #shepe(Q,K,V)=[B x Seq x D/num_heads]*num_heads

        output_per_head=[]
        attention_weights_per_head=[]

        for Q_,K_,V_ in zip(Q,K,V):
            output,attention_weights=self.scaled_dot_product_attention(Q_,K_,V_)

            output_per_head.append(output)
            attention_weights_per_head.append(attention_weights)
        

        
        output=torch.cat(output_per_head,-1)
        #shape(output)=[B x seq x D ]
        #shape(attention_weights)=[B x seq x seq]
        attention_weights=torch.stack(attention_weights_per_head).permute(1,0,2,3)       
        #shape(attention_weights)=[B x num_heads x seq x seq]

        projection=self.dropout(self.mha_linear(output))

        return projection,attention_weights


    