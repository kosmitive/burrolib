"""Represents an interface for one agent. Different implementations
can be used as this will be used inside the simulation to drive the
different goods."""
class Agent:

    def getOutgoingOrders(self, pos, supply, deliveries):
        return 30