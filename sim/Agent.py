"""Represents an interface for one agent. Different implementations
can be used as this will be used inside the simulation to drive the
different goods."""
class Agent:

    def registerIncomingOrder(self, n):
        return

    def registerIncomingDeliveries(self, n):
        return

    def process(self):
        return

    def getOutgoingOrders(self):
        return