import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class SparseDeepUnrolling(nn.Module):
    ## This section is largely inspired by the code from the paper: "Stable and Interpretable Unrolled Dictionary Learning Deep Networks"
    ## https://github.com/btolooshams/stable-interpretable-unrolled-dl/blob/master/src/model.py
    def __init__(self, W = None, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        
        self.D = kwargs.get('embedding_dim', 512)
        self.P = kwargs.get('dict_size', 1000)
        self.T = kwargs.get('num_layers', 10)

        
        self.lam = kwargs.get('lam', 0.12)
        self.step = kwargs.get('step', 0.1)

        if W is None:
            self.W = F.normalize(nn.Parameter(torch.randn((self.D, self.P), device=self.device)), p=2, dim=0)
        else:
            self.W = F.normalize(nn.Parameter(W.to(self.device)), p=2, dim=0)

    def normalize(self):
        self.W.data = F.normalize(self.W.data, p=2, dim=0)

    def get_params(self):
        return self.state_dict(keep_vars=True)

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        zhat = torch.zeros((batch_size, self.P, 1), device=self.device)
        IplusWTW = torch.eye(self.P, device=self.device) - self.step * torch.matmul(torch.t(self.W), self.W)
        WTx = self.step * torch.matmul(torch.t(self.W), x)
        #print(IplusWTW.shape, zhat.shape, WTx.shape)
        for _ in range(self.T):
            zhat = F.relu(torch.matmul(IplusWTW, zhat) + WTx - self.lam * self.step)

        xhat = torch.matmul(self.W, zhat)

        return xhat, zhat


if __name__ == '__main__':
    params = {
        'embedding_dim': 512,
        'dict_size': 1000,
        'num_layers': 10,
        'lam': 0.12,
        'step': 0.1
    }
    model = SparseDeepUnrolling(**params)

    print(model.W.shape)
    x = torch.randn((10, 512, 1))
    a = model(x)