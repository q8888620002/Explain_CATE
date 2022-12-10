import torch
import torch.nn as nn
import torch.optim as optim



class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    Args:
      append:
      mask_size:
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, m):
        out = x * m
        if x.shape[1] == m.shape[1]:
            out = torch.cat([out, m], dim=1)
        return out
    

class Cate(nn.Module):
    
    def __init__(self, model, masklayer):
        super().__init__()
        
        self.masklayer = masklayer
        self.model = model
        
    def fit(self, x, x_mask, y , w):
        '''
        Training CATE model with x*mask + mask
        '''
        #### TODO sample random masking
        
        x = self.masklayer(x, x_mask)
        
        self.model.fit(x,y,w)
        
    def predict(self, x_test):
        '''
        Predict Phe for X_test
        '''
        #mask = torch.ones(x_test.size()[0],x_test.size()[1])
        #x_test = torch.cat([x_test, mask], dim=1)
        
        return self.model.predict(x_test)
        