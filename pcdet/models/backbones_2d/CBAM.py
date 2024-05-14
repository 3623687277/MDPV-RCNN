###空间注意力机制和通道注意力机制
import torch
from torch import nn



class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16): 
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        self.fc=nn.Sequential(nn.Linear(in_channel, in_channel // ratio, bias=False),
                              nn.ReLU(inplace=True),
                              nn.Linear(in_channel // ratio, in_channel, bias=False)) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        batch,channel,H,W=x.size() 
        max_pool_out=self.max_pool(x).view([batch,channel]) 
        avg_pool_out=self.avg_pool(x).view([batch,channel]) 
        max_fc_out=self.fc(max_pool_out)
        avg_fc_out=self.fc(avg_pool_out) 
        out=max_fc_out+avg_fc_out 
        out=self.sigmoid(out).view(batch,channel,1,1) 
        out=out*x
        return out
    

##空间注意力机制
class SpaceAttention(nn.Module):
    def __init__(self,kernel_size=7): 
        super(SpaceAttention, self).__init__()
        self.conv=nn.Conv2d(2,1,kernel_size,1,padding=7//2,bias=False)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        batch,channel,H,W=x.size() 
        max_pool_out,_=torch.max(x,dim=1,keepdim=True) 
        avg_pool_out=torch.mean(x,dim=1,keepdim=True) 
        pool_out=torch.cat([max_pool_out,avg_pool_out],dim=1) 
        out=self.conv(pool_out) 
        out=self.sigmoid(out)  
        return out*x

##实现通道注意和空间注意力的结合应用
class Cbam(nn.Module):
    def __init__(self,channel,reduction=16,kernel_size=7):
        super(Cbam, self).__init__()
        
        self.channel_attention=ChannelAttention(channel,reduction)
        
        self.space_attention=SpaceAttention(kernel_size)

    
    def forward(self, x):
        x=self.channel_attention(x)
        x=self.space_attention(x)
        return x
    
if __name__=='__main__':
    model=Cbam(512) #512表示通道，其他的都有默认值
    print(model)
    input=torch.ones([2,512,26,26])
    output=model(input)
    print(output.size())


        
