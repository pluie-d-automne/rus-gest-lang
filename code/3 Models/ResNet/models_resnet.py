import torch 
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self,in_features=64,out_features=64,stride=[1,1],down_sample=False):
        # stride : list 
        # the value at corresponding indices are the strides of corresponding layers in a residual block
        
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv3d(in_features,out_features,3,stride[0],padding=1,bias=False) #weight layer
        self.bn1 = nn.BatchNorm3d(out_features) #weight layer
        
        self.relu = nn.ReLU(True) #relu
        
        self.conv2 = nn.Conv3d(out_features,out_features,3,stride[1],padding=1,bias=False) #weight layer
        self.bn2 = nn.BatchNorm3d(out_features) #weight layer

        self.down_sample = down_sample
        if down_sample:
            self.downsample = nn.Sequential(
                    nn.Conv3d(in_features,out_features,1,2,bias=False),
                    nn.BatchNorm3d(out_features)
                )

    def forward(self,x):
        x0=x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:
            x0 = self.downsample(x0)  
        x = x + x0    # F(x)+x
        x= self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,inFeatures=64,outFeatures=64,kSize=[1,3,1],stride=[1,2,1],
    dn_sample=False,dnSample_stride=1) -> None:
        super(Bottleneck,self).__init__()
        
        
        self.conv1 = nn.Conv3d(inFeatures,outFeatures,kSize[0],stride[0],bias=False)
        self.bn1 = nn.BatchNorm3d(outFeatures)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv3d(outFeatures,outFeatures,kSize[1],stride[1],padding=1,bias=False)
        self.bn2 = nn.BatchNorm3d(outFeatures)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv3d(outFeatures,outFeatures*4,kSize[2],stride[2],bias=False)
        self.bn3 = nn.BatchNorm3d(outFeatures*4) 
        self.relu3 = nn.ReLU(True)
        

        self.ds = dn_sample
        if dn_sample:
            self.downSample = nn.Sequential(
                nn.Conv3d(inFeatures,outFeatures*4,1,stride=dnSample_stride,bias=False),
                nn.BatchNorm3d(outFeatures*4)            
            )
        
    
    def forward(self,x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        if self.ds:
            x0 = self.downSample(x0)
        x = x+x0
        return x


class ResNet(nn.Module):

    def __init__(self,in_channels=3,num_residual_block=[3,4,6,3],num_class=1000,block_type='normal'):
        super(ResNet,self).__init__()

        self.conv1 = nn.Conv3d(in_channels,64,(1,7,7),(1,2,2,),(1,3,3),bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool3d(3,2,1)

        if block_type.lower() == 'bottleneck':    
            self.resnet,outchannels = self.__bottlenecks(num_residual_block)
        else:
            self.resnet,outchannels = self.__layers(num_residual_block)
    

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(in_features=outchannels,out_features=num_class,bias=True)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
    
    def __layers(self,num_residual_block):
        layer=[]
        layer += [BasicBlock()]*num_residual_block[0]
        inchannels=64
        for numOFlayers in num_residual_block[1:]:
            stride = [2,1] #updating the stride, the first layer of residual block
            # will have a stride of two and the 2nd layer of the residual block have 
            # a stride of 1
            downsample=True
            outchannels = inchannels*2
            for _ in range(numOFlayers):
                layer.append(BasicBlock(inchannels,outchannels,stride,down_sample=downsample))
                inchannels = outchannels
                downsample = False 
                stride=[1,1]
            
        return nn.Sequential(*layer),outchannels

    
    def __bottlenecks(self,numres):
        
        layer=[]
        
        stride = [1,1,1]
        dnStride=1
        inchan = 64
        for i,numOFlayers in enumerate(numres):
            dn_sample = True
            outchan = 64*(2**i)

            for _ in range(numOFlayers):
                layer+=[ 
                    Bottleneck(inchan,outchan,stride=stride,
                    dn_sample=dn_sample,dnSample_stride=dnStride)
                ]
                inchan = outchan*4
                dn_sample = False
                stride = [1,1,1]   
            dn_sample=True 
            stride = [1,2,1]
            dnStride=2
            

        return nn.Sequential(*layer),inchan
    

def  resnet18(**kwargs):
    return ResNet(num_residual_block=[2,2,2,2],**kwargs)

def resnet34(**kwargs):
    return ResNet(num_residual_block=[3,4,6,3],**kwargs)

def resnet50(**kwargs):
    return ResNet(num_residual_block=[3,4,6,3],block_type='bottleneck',**kwargs)

def resnet101(**kwargs):
    return ResNet(num_residual_block=[3,4,23,3],block_type='bottleneck',**kwargs)

def resnet152(**kwargs):
    return ResNet(num_residual_block=[3,8,36,3],block_type='bottleneck',**kwargs)
