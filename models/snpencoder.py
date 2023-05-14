import torch.nn as nn 
import torch.nn.functional as F

class SNPEncoder(nn.Module):
    """
    MLP
    """
    def __init__(self, x_dim, out_dim, fc_dims, use_bn = True):
        """
            fc_dims -- A list of fully connected layers
            p -- Dropout
        """
        super(SNPEncoder, self).__init__()
        self.x_dim = x_dim
        self.use_bn = use_bn
        self.enc_shape = fc_dims
        self.enc = []
        for i in range(len(self.enc_shape)):
            if i == 0: # set intial data fc from x -> i
                self.enc.append(nn.Linear(self.x_dim, self.enc_shape[i]))
            else: # fc from i-1 -> i
                self.enc.append(nn.Linear(self.enc_shape[i-1], self.enc_shape[i]))

            if self.use_bn:
                self.enc.append(nn.BatchNorm1d(self.enc_shape[i]))
        self.enc = nn.ModuleList(self.enc)
        self.out = nn.Linear(fc_dims[-1], out_dim)

    def forward(self, x):
        # encode x's until shared layer
        for l in self.enc:
            x = l(x)
            if isinstance(l, nn.BatchNorm1d):
                x = F.relu(x) 
        z = self.out(x)
        z = F.normalize(z, dim = 1)
        
        return z