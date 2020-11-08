import torch
import torch.nn.functional as F
from torch import distributions
from torch import nn
from typing import Tuple

from burro import BasePolicyModel
from burro import MLP


class GumbelPolicy(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_size: Tuple[int] = (64, 64,), nonlin: str = 'relu'):
        super(GumbelPolicy, self).__init__()
        self.net = MLP(in_dim, out_dim, hidden_size, nonlin)
        self.temperature = nn.Parameter(torch.ones(1, 1)) # * torch.tensor(std).log())

    def forward(self, x):
        logits = self.net(x)

        return logits, self.temperature


class CategoricalPolicyModel(BasePolicyModel):

    def __init__(self, io_order_dim: int = 3, max_order_size: int = 10, p_hidden_size: Tuple[int] = (64, 64,),
                 vf_hidden_size: Tuple[int] = (64, 64,), p_lr: float = 1e-3, vf_lr: float = 1e-3):
        super(CategoricalPolicyModel, self).__init__()
        self.max_order_size = max_order_size
        self.policy_net = MLP(in_dim=io_order_dim, out_dim=max_order_size, hidden_size=p_hidden_size, nonlin='elu')
        self.vf_net = MLP(in_dim=io_order_dim, out_dim=1, hidden_size=vf_hidden_size, nonlin='elu')
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=p_lr)
        self.vf_optim = torch.optim.Adam(self.vf_net.parameters(), lr=vf_lr)

    def act(self, state):
        logits = self.policy_net(state)
        out_orders = distributions.Categorical(logits=logits).sample()
        return out_orders.detach().numpy()

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


if __name__ == '__main__':
    lol = CategoricalPolicyModel()

    x = torch.tensor([5., 4., 2.])

    a = lol.act(x)
    print(a)