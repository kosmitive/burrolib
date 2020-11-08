import torch
from torch import distributions
from copy import deepcopy

from burro.agents import Agent
from burro import CategoricalPolicyModel
from burro import discount


class CategoricalAgent(Agent):

    def __init__(self, policy_model: CategoricalPolicyModel, gamma: float = 0.99,
                 batch_size: int = 100, vf_epochs: int = 25):
        super(CategoricalAgent, self).__init__()
        self.policy_model = policy_model
        self.max_order_size = self.policy_model.max_order_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.states_batch = torch.empty((self.batch_size, 3))
        self.reward_batch = torch.zeros((self.batch_size, 1))
        self.oorders_batch = torch.empty((self.batch_size, 1))
        self.batch_fill_count = 0
        self.vf_epochs = vf_epochs

    def clone(self):
        return CategoricalAgent(deepcopy(self.policy_model), gamma=self.gamma, batch_size=self.batch_size,
                                vf_epochs=self.vf_epochs)

    def get_outgoing_orders(self, pos, clen, stock, iorders, oorders, costs):
        state = torch.stack([torch.tensor(stock, dtype=torch.float32), torch.tensor(iorders, dtype=torch.float32),
                             torch.tensor(oorders, dtype=torch.float32)])
        new_oorders = self.policy_model.act(state)
        self.oorders_batch[self.batch_fill_count] = torch.from_numpy(new_oorders)
        self.states_batch[self.batch_fill_count] = state
        self.reward_batch[self.batch_fill_count] = -costs # (torch.eye(self.max_order_size)[iorders] - train_orders).mean()
        self.batch_fill_count += 1
        if self.batch_fill_count >= self.batch_size:
            discounted_reward = discount(self.reward_batch.numpy(), self.gamma)
            discounted_reward = torch.cat([torch.tensor(dr) for dr in discounted_reward])
            self.batch_fill_count = 0
            self.update_vf(self.states_batch, discounted_reward)
            advantage = discounted_reward - self.policy_model.vf_net(self.states_batch)
            self.update_policy(self.states_batch, advantage, self.oorders_batch)

        return new_oorders

    def update_policy(self, states, advantage, oorders):
        logits = self.policy_model.policy_net(states)
        log_probs = torch.distributions.Categorical(logits=logits).log_prob(oorders)
        loss = - (advantage * log_probs).mean()
        self.policy_model.policy_optim.zero_grad()
        loss.backward()
        self.policy_model.policy_optim.step()

    def update_vf(self, states, reward):
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.vf_epochs):
            values = self.policy_model.vf_net(states)
            loss = loss_fn(reward[:, None], values).mean(0)
            self.policy_model.vf_optim.zero_grad()
            loss.backward()
            self.policy_model.vf_optim.step()
