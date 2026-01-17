import torch
import torch.nn as nn




class Efficient_Channel_Attention(nn.Module):
    
    def __init__(self, channel, k_size=3):
        super(Efficient_Channel_Attention, self).__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.conv=nn.Conv1d(1,1,kernel_size=k_size, padding=((k_size-1)//2), bias=False)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x): 
        y=self.avg_pool(x) 
        y=y.squeeze(-1).permute(0,2,1) 
        y=self.conv(y)
        y=self.sigmoid(y) 
        y=y.permute(0,2,1).unsqueeze(-1) 
        
        return x*y.expand_as(x) # Multiply elementwise





