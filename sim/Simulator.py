from sim.DiscretePoissonProcess import DiscretePoissonProcess
from sim.Agent import Agent


class Simulator:
    """This class represents a simple simulator. It uses the defined
    agent interface to run the simulation."""

    def __init__(self, agents):
        self.N = 4
        self.outstanding = 0
        self.current_step = 0
        self.process = DiscretePoissonProcess(2.5)
        self.orders = [0] * (self.N + 1)
        self.products = [0] * (self.N + 1)

        assert isinstance(agents, Agent)
        self.agents = [agents] * self.N

    def one_step(self):
        """This method performs one discrete step in the simulator."""

        # place the end consumer order
        self.orders[-1] = self.process.get_discrete_increase()
        self.outstanding += self.orders[-1]

        # order flow
        [self.agents[k].registerIncomingOrder(k + 1) for k in range(self.N)]

        # product flow
        [self.agents[k].registerIncomingDeliveries(self.products[k]) for k in range(self.N)]

        # remove the end consumer product from outstanding
        self.outstanding -= self.products[-1]

        # let all agents do their calculations
        [self.agents[k].process() for k in range(self.N)]

        # place the desired orders in the array
        self.orders[::-1] = [self.agents[k].getOutgoingOrders() for k in range(self.N)]

    def mult_steps(self, steps):
        """Performs multiple steps using the one step method."""

        [self.one_step() for k in range(steps)]