from burro.agents.agent import Agent


class RefillAgent(Agent):

    def __init__(self, amount, limit):
        self.amount = amount
        self.limit = limit

    def experience(self, state, action, reward, nxt_state, done): pass

    def train(self): pass

    def sync(self): pass

    def act(self, state):
        supply, orders, transported = list(state)
        return self.amount if supply < self.limit else 0
