# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:59:32 2021

@author: user
"""

import numpy as np
from Model_architecture import EEGfuseNet_Channel_32
import torch
EEG_data=np.random.randn(128,1,32,384) # (batch_size,1,EEG channel,Fs)--Preprocessed EEG 1s segment. In this demo we use noisy signal for convenience.
EEG_data = EEG_data.astype('float32')
EEG_data = torch.from_numpy(EEG_data)
model=EEGfuseNet_Channel_32(16,1,1,384).cuda(0)
model.load_state_dict(torch.load('Pretrained_model.pkl'))
output_signal,deep_feature=model(EEG_data.cuda(0))