import torch
from torch import distributions
from copy import deepcopy

from burro.agents.agent import Agent
from burro.policies.categorical_policy import CategoricalPolicyModel
from burro.util.calc import discount


class CategoricalAgent(Agent):

    def __init__(self, policy_model: CategoricalPolicyModel, gamma: float = 0.99,
                 batch_size: int = 100, vf_epochs: int = 25):
        super(CategoricalAgent, self).__init__()
        self.policy_model = deepcopy(policy_model)
        self.max_order_size = self.policy_model.max_order_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.states_batch = torch.empty((self.batch_size, 3)).double()
        self.reward_batch = torch.zeros((self.batch_size, 1)).double()
        self.nxt_actions_batch = torch.empty((self.batch_size, 1)).double()
        self.batch_fill_count = 0
        self.vf_epochs = vf_epochs
        self.batch_filled = False

    def _clone(self):
        return CategoricalAgent(deepcopy(self.policy_model), gamma=self.gamma, batch_size=self.batch_size,
                                vf_epochs=self.vf_epochs)

    def _fill_batch(self, state, nxt_action, reward):
        self.states_batch[self.batch_fill_count] = state
        self.nxt_actions_batch[self.batch_fill_count] = nxt_action
        self.reward_batch[self.batch_fill_count] = -reward

    def experience(self, state, action, reward, nxt_state, done):
        nxt_action = self.policy_model.act(state)
        self._fill_batch(torch.from_numpy(state), torch.from_numpy(nxt_action), -reward)
        self.batch_fill_count += 1
        if self.batch_fill_count >= self.batch_size:
            self.batch_filled = True

    def act(self, state):
        return self.policy_model.act(state)

    def sync(self):
        pass

    def train(self):
        if self.batch_filled:
            discounted_reward = discount(self.reward_batch.numpy(), self.gamma)
            discounted_reward = torch.cat([torch.tensor(dr) for dr in discounted_reward])
            # Train value function net
            self.update_vf(self.states_batch, discounted_reward)

            # Compute quantities for policy optimization
            advantage = (discounted_reward - self.policy_model.vf_net(self.states_batch)).detach()
            logits = self.policy_model.policy_net(self.states_batch)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(self.nxt_actions_batch)
            loss = - (advantage * log_probs).mean()

            # Update policy network
            self.policy_model.policy_optim.zero_grad()
            loss.backward()
            self.policy_model.policy_optim.step()

            # Reset batch filled criterion
            self.batch_filled = False
            self.batch_fill_count = 0

    def update_vf(self, states, reward):
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.vf_epochs):
            values = self.policy_model.vf_net(states)
            loss = loss_fn(reward[:, None], values).mean(0)
            self.policy_model.vf_optim.zero_grad()
            loss.backward()
            self.policy_model.vf_optim.step()
