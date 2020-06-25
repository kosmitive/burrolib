from src.agents.agent import Agent


class RefillAgent(Agent):

    def __init__(self, amount, limit):
        self.amount = amount
        self.limit = limit

    def clone(self):
        return RefillAgent(self.amount, self.limit)

    def get_outgoing_orders(self, pos, clen, stock, iorders, oorders, costs):
        return self.amount if stock < self.limit else 0
