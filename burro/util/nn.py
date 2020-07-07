import torch
from typing import Tuple


class MLP(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_size: Tuple[int] = (64,), nonlin: str = 'relu', **kwargs):
        super(MLP, self).__init__()
        nlist = dict(relu=torch.nn.ReLU(), tanh=torch.nn.Tanh(),
                     sigmoid=torch.nn.Sigmoid(), softplus=torch.nn.Softplus(), lrelu=torch.nn.LeakyReLU(),
                     elu=torch.nn.ELU())

        self.nonlin = nlist[nonlin]
        self.net = torch.nn.ModuleList([torch.nn.Linear(in_dim, hidden_size[0]),
                                  self.nonlin])
        if len(hidden_size) > 1:
            for i in range(len(hidden_size) - 1):
                self.net.append(torch.nn.Linear(hidden_size[i], hidden_size[i + 1]))
        self.out = torch.nn.Linear(hidden_size[-1], out_dim)

    def forward(self, x, **kwargs):
        for _, layer in enumerate(self.net):
            x = layer(x)

        return self.out(x)
