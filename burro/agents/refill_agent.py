from burro.agents.agent import Agent


class RefillAgent(Agent):

    def __init__(self, amount, limit):
        self.amount = amount
        self.limit = limit

    def act(self, *args):
        supply, orders, transported = args
        return self.amount if supply < self.limit else 0
