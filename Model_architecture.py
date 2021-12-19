# -*- coding: utf-8 -*-
"""
@author: RushuangZhou
"""
#Please cite our paper as:
#Z. Liang, R. Zhou, L. Zhang, L. Li, G. Huang, Z. Zhang, and S. Ishii, EEGFuseNet: Hybrid Unsupervised Deep Feature Characterization and Fusion for High-Dimensional EEG With an #Application to Emotion Recognition, IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29, pp. 1913-1925, 2021.

import torch.nn as nn
import torch.nn.functional as F
import torch
# hidden_dim: hidden dim of BI-GRU, in our study, hidden_dim=16;
# n_layer : the number of layers of BI-GRU,in our study, n_layer=1;
# n_filters: hyperparameter for controlling the number of filters of each convolution layers,in our study, n_filters=1;
# input_size: sampling rate of 1s EEG signal (in our study, the size of the input 1s EEG signal is 32 × 384 (Channel × Time), so the sampling rate is 384hz);
class EEGfuseNet_Channel_32(nn.Module):
        def __init__(self,hidden_dim,n_layer,n_filters,input_size):  
            super(EEGfuseNet_Channel_32, self).__init__()
            ## Channel 32 network for DEAP,HCI dataset.
            # conventional convolution for time dim
            self.conv1 = nn.Conv2d(1, int(16*n_filters), (1, int(input_size/2+1)),stride = 1, padding = (0,int(input_size/4)))
            self.batchNorm1 =nn.BatchNorm2d(int(16*n_filters), False)
            self.n_filters=n_filters
            self.hidden_dim=hidden_dim
            self.length=input_size/32
            # spatial convolution for channel dim
            self.depthwiseconv2 = nn.Conv2d(int(16*n_filters),int(32*n_filters),(32,1),padding = 0) ## 32 channel EEG signal from DEAP,HCI dataset, so the kernel length here is 32
            self.batchNorm2 = nn.BatchNorm2d(int(32*n_filters),False)
            self.pooling1 = nn.MaxPool2d((1,4),return_indices =True)
        
            # depthwise separable  convolutions 
            self.separa1conv3 = nn.Conv2d(int(32*n_filters),int(32*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(32*n_filters)) #(32, 1, T/5)
            self.separa2conv4 = nn.Conv2d(int(32*n_filters),int(16*n_filters),1) 
            self.batchNorm3 = nn.BatchNorm2d(int(16*n_filters),False)
            self.pooling2 = nn.MaxPool2d((1,8), return_indices =True)
            self.dropout1 = nn.Dropout(p=0.25)
            self.dropout2 = nn.Dropout(p=0.25)
            self.dropout3 = nn.Dropout(p=0.25)
            self.dropout4 = nn.Dropout(p=0.25)
            self.fc1 = nn.Linear(int(16*n_filters),int(16*n_filters))
            self.fc2 = nn.Linear(int(hidden_dim*2*n_filters),int(hidden_dim*n_filters))
            self.fc3 = nn.Linear(int(hidden_dim*n_filters),int(hidden_dim*2*n_filters))
            self.fc4 = nn.Linear(int(2*16*n_filters),int(16*n_filters))
            # GRU
            self.gru_en = nn.GRU(int(16*n_filters),int(hidden_dim*n_filters),n_layer,batch_first=True,bidirectional=True)
            self.gru_de = nn.GRU(int(2*hidden_dim*n_filters),int(16*n_filters),n_layer,batch_first=True,bidirectional=True)
            self.lstm = nn.LSTM(int(16*n_filters),int(hidden_dim*n_filters),n_layer,batch_first=True,bidirectional=True)
            # deconventional
            self.unpooling2 = nn.MaxUnpool2d((1, 8))
            self.batchnorm4 = nn.BatchNorm2d(int(32*n_filters),False)
            self.desepara2conv4 =  nn.ConvTranspose2d(int(16*n_filters),int(32*n_filters),1)
            self.desepara1conv3 =  nn.ConvTranspose2d(int(32*n_filters),int(32*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(32*n_filters))
        
            #de spatial convolution for channel dim
            self.unpooling1 = nn.MaxUnpool2d((1, 4))
            self.batchnorm5 = nn.BatchNorm2d(int(16*n_filters), False)#
            self.dedepthsepara1conv3 = nn.ConvTranspose2d(int(32*n_filters),int(16*n_filters), (32, 1), stride = 1, padding = 0)
        
            #de spatial convolution for channel dim
            self.deconv1 = nn.ConvTranspose2d(int(16*n_filters), 1, (1, int(input_size/2+1)), stride = 1, padding = (0,int(input_size/4)))
        def forward(self,x):
            # encoder
            x = self.conv1(x)
            x = self.batchNorm1(x)
            # Layer 2
            x = self.depthwiseconv2(x)
            x = self.batchNorm2(x)       
            x = F.elu(x)
            x,idx2 = self.pooling1(x)  # get data and their index after pooling 
            x = self.dropout1(x)
            # Layer 3
            x = self.separa1conv3(x)       
            x = self.separa2conv4(x)       
            x = self.batchNorm3(x)
            x = F.elu(x)
            x,idx3 = self.pooling2(x)
        
            # Layer 4：FC Layer
            x =x.permute(0,3,2,1)
            x =x[:,:,-1,:,]
            x =self.fc1(x)
            x =F.elu(x)
            out,_=self.gru_en(x)
            x=out
            x = self.fc2(x)
            
            code=x.reshape((x.shape[0],int(16*self.n_filters)*int(self.length))) # code representation for clustering
            
            #decoder
            x = self.fc3(x)
            out,_=self.gru_de(x)
            x = out
            x = self.fc4(x)
            x = F.elu(x)
            x = x.reshape((x.shape[0],x.shape[1],1,x.shape[2]))
            x = x.permute(0,3,2,1)
            x = self.unpooling2(x, idx3)

            x = self.desepara2conv4(x)
            x = self.desepara1conv3(x)
            x = self.batchnorm4(x)
            x = self.dropout4(x)

            # Layer 3
            x = F.elu(x)   
            x = self.unpooling1(x, idx2)      
            x = self.dedepthsepara1conv3(x)
            x = self.batchnorm5(x)
            # Layer 4
            x = self.deconv1(x)
            return x,code
        
class EEGfuseNet_Channel_62(nn.Module):
        ## Channel 62 network for SEED dataset.
        def __init__(self,hidden_dim,n_layer,n_filters,input_size):
            super(EEGfuseNet_Channel_62, self).__init__()
            self.conv1 = nn.Conv2d(1, int(16*n_filters), (1, int(input_size/2+1)),stride = 1, padding = (0,int(input_size/4))) #(16, C, T)
            self.batchNorm1 =nn.BatchNorm2d(int(16*n_filters), False)
            self.n_filters=n_filters
            self.hidden_dim=hidden_dim
            self.length=input_size/32
            self.depthwiseconv2 = nn.Conv2d(int(16*n_filters),int(32*n_filters),(62,1),padding = 0)  ## 62 channel EEG signal from SEED dataset, so the kernel length here is 62
            self.batchNorm2 = nn.BatchNorm2d(int(32*n_filters),False)
            self.pooling1 = nn.MaxPool2d((1,4),return_indices =True)
            self.separa1conv3 = nn.Conv2d(int(32*n_filters),int(32*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(32*n_filters)) #(32, 1, T/5)
            self.separa2conv4 = nn.Conv2d(int(32*n_filters),int(16*n_filters),1)
            self.batchNorm3 = nn.BatchNorm2d(int(16*n_filters),False)
            self.pooling2 = nn.MaxPool2d((1,8), return_indices =True)
            self.dropout1 = nn.Dropout(p=0.25)
            self.dropout2 = nn.Dropout(p=0.25)
            self.dropout3 = nn.Dropout(p=0.25)
            self.dropout4 = nn.Dropout(p=0.25)
            self.fc1 = nn.Linear(int(16*n_filters),int(16*n_filters))
            self.fc2 = nn.Linear(int(hidden_dim*2*n_filters),int(hidden_dim*n_filters))
            self.fc3 = nn.Linear(int(hidden_dim*n_filters),int(hidden_dim*2*n_filters))
            self.fc4 = nn.Linear(int(2*16*n_filters),int(16*n_filters))
            self.gru_en = nn.GRU(int(16*n_filters),int(hidden_dim*n_filters),n_layer,batch_first=True,bidirectional=True)
            self.gru_de = nn.GRU(int(2*hidden_dim*n_filters),int(16*n_filters),n_layer,batch_first=True,bidirectional=True)
            self.lstm = nn.LSTM(16*n_filters,hidden_dim*n_filters,n_layer,batch_first=True,bidirectional=True)
            self.unpooling2 = nn.MaxUnpool2d((1, 8))
            self.batchnorm4 = nn.BatchNorm2d(int(32*n_filters),False)
            self.desepara2conv4 =  nn.ConvTranspose2d(int(16*n_filters),int(32*n_filters),1)
            self.desepara1conv3 =  nn.ConvTranspose2d(int(32*n_filters),int(32*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(32*n_filters))
            self.unpooling1 = nn.MaxUnpool2d((1, 4))
            self.batchnorm5 = nn.BatchNorm2d(int(16*n_filters), False)
            self.dedepthsepara1conv3 = nn.ConvTranspose2d(int(32*n_filters),int(16*n_filters), (62, 1), stride = 1, padding = 0)
            self.deconv1 = nn.ConvTranspose2d(int(16*n_filters), 1, (1, int(input_size/2+1)), stride = 1, padding = (0,int(input_size/4)))
        def forward(self,x):
            # encoder
            x = self.conv1(x)
            x = self.batchNorm1(x)
            # Layer 2
            x = self.depthwiseconv2(x)
            x = self.batchNorm2(x)       
            x = F.elu(x)
            x,idx2 = self.pooling1(x)  # get data and their index after pooling 
            x = self.dropout1(x)
            # Layer 3
            x = self.separa1conv3(x)       
            x = self.separa2conv4(x)       
            x = self.batchNorm3(x)
            x = F.elu(x)
            x,idx3 = self.pooling2(x)
        
            # Layer 4：FC Layer
            x =x.permute(0,3,2,1)
            x =x[:,:,-1,:,]
            x =self.fc1(x)
            x =F.elu(x)
            out,_=self.gru_en(x)
            x=out
            x = self.fc2(x)
            
            code=x.reshape((x.shape[0],int(16*self.n_filters)*int(self.length)))
            
            #decoder
            x = self.fc3(x)
            out,_=self.gru_de(x)
            x = out
            x = self.fc4(x)
            x = F.elu(x)
            x = x.reshape((x.shape[0],x.shape[1],1,x.shape[2]))
            x = x.permute(0,3,2,1)
            x = self.unpooling2(x, idx3)

            x = self.desepara2conv4(x)
            x = self.desepara1conv3(x)
            x = self.batchnorm4(x)
            x = self.dropout4(x)

            # Layer 3
            x = F.elu(x)   
            x = self.unpooling1(x, idx2)      
            x = self.dedepthsepara1conv3(x)
            x = self.batchnorm5(x)
            # Layer 4
            x = self.deconv1(x)
            return x,code

class Discriminator_Channel_32(nn.Module):
        def __init__(self,n_layer,n_filters,input_size):
            super(Discriminator_Channel_32, self).__init__()
            self.conv1 = nn.Conv2d(1,int(8*n_filters),(1, int(input_size/2+1)),stride = 1, padding = (0,int(input_size/4))) #(16, C, T)
            self.batchNorm1 =nn.BatchNorm2d(8*n_filters, False)
            self.length=input_size/32
            # Layer 2：spatial convolution for channel dim
            self.depthwiseconv2 = nn.Conv2d(int(8*n_filters),int(16*n_filters),(32,1),padding = 0)  
            self.batchNorm2 = nn.BatchNorm2d(int(16*n_filters),False)
            self.pooling1 = nn.MaxPool2d((1,4),return_indices=False)
            self.separa1conv3 = nn.Conv2d(int(16*n_filters),int(16*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(16*n_filters))
            self.separa2conv4 = nn.Conv2d(int(16*n_filters),int(8*n_filters),1) 
            self.batchNorm3 = nn.BatchNorm2d(int(8*n_filters),False)
            self.pooling2 = nn.MaxPool2d((1,8), return_indices=False)
            self.fc1 = nn.Linear(int(self.length*8),1)
        def forward(self,x):
            # encoder
            x = self.conv1(x)
            x = self.batchNorm1(x)
            # Layer 2
            x = self.depthwiseconv2(x)
            x = self.batchNorm2(x)       
            x = F.elu(x)
            x = self.pooling1(x)  # get data and their index after pooling 
            # Layer 3
            x = self.separa1conv3(x)       
            x = self.separa2conv4(x)       
            x = self.batchNorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
        
            # Layer 4：FC Layer
            x = x.reshape((x.shape[0],int(self.length*8)))
            x = self.fc1(x)
            x=torch.sigmoid(x)
            return x

class Discriminator_Channel_62(nn.Module):
        def __init__(self,n_layer,n_filters,input_size):
            super(Discriminator_Channel_62, self).__init__()
            self.conv1 = nn.Conv2d(1,int(8*n_filters),(1, int(input_size/2+1)),stride = 1, padding = (0,int(input_size/4)))
            self.batchNorm1 =nn.BatchNorm2d(8*n_filters, False)
            self.length=input_size/32
            # Layer 2：spatial convolution for channel dim
            self.depthwiseconv2 = nn.Conv2d(int(8*n_filters),int(16*n_filters),(62,1),padding = 0)
            self.batchNorm2 = nn.BatchNorm2d(int(16*n_filters),False)
            self.pooling1 = nn.MaxPool2d((1,4),return_indices=False)
            self.separa1conv3 = nn.Conv2d(int(16*n_filters),int(16*n_filters),(1, int(input_size/8+1)),stride=1,padding=(0,int(input_size/16)),groups=int(16*n_filters))
            self.separa2conv4 = nn.Conv2d(int(16*n_filters),int(8*n_filters),1) 
            self.batchNorm3 = nn.BatchNorm2d(int(8*n_filters),False)
            self.pooling2 = nn.MaxPool2d((1,8), return_indices=False)
            self.fc1 = nn.Linear(int(self.length*8),1)
        def forward(self,x):
            # encoder
            x = self.conv1(x)
            x = self.batchNorm1(x)
            # Layer 2
            x = self.depthwiseconv2(x)
            x = self.batchNorm2(x)       
            x = F.elu(x)
            x = self.pooling1(x)
            # Layer 3
            x = self.separa1conv3(x)       
            x = self.separa2conv4(x)       
            x = self.batchNorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
        
            # Layer 4：FC Layer
            x = x.reshape((x.shape[0],int(self.length*8)))
            x = self.fc1(x)
            x=torch.sigmoid(x)
            return x       
