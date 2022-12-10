import torch
import torch.nn as nn
import torch.optim as optim
from utilities import *


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, m):
        out = x * m
        if x.shape[1] == m.shape[1]:
            out = torch.cat([out, m], dim=1)
        return out
    

class Cate(nn.Module):
    
    def __init__(self, model, masklayer, device):
        super().__init__()
        
        self.masklayer = masklayer
        self.model = model
        self.device = device

    def fit(self, x_train, y_train , w_train, epoches):
        '''
        Training CATE model with x*mask + mask
        '''

        for i in range(epoches):
            random_maskes = generate_maskes(x_train).to(self.device)
            x_train_w_mask = self.masklayer(x_train, random_maskes)
            self.model.fit(x_train_w_mask,y_train,w_train)
        
    def predict(self, x_test, test_mask):
        '''
        Predict Phe for X_test
        '''
        x_test = self.masklayer(x_test, test_mask)
        
        return self.model.predict(x_test)
        