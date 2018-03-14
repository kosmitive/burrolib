import numpy as np
from python.sim.DiscretePoissonProcess import DiscretePoissonProcess
from python.sim.Agent import Agent


class Simulator:
    """This class represents a simple simulator. It uses the defined
    agent interface to run the simulation."""

    def __init__(self, agents, N = 10):
        self.N = N
        self.outstanding = 0
        self.current_step = 0

        # define a dirichlet poisson process
        self.process = DiscretePoissonProcess(2.5)

        # define the data structure
        self.current_supply = 15 * np.ones([N], np.int32)
        self.orders = 5 * np.ones([N + 1], np.int32)
        self.running_deliveries = 5 * np.ones([N + 1], np.int32)
        self.cost = np.zeros([N + 1])

        assert isinstance(agents, list)
        self.agents = agents

    def one_step(self):
        """This method performs one discrete step in the simulator."""

        self.orders -= self.running_deliveries
        self.current_supply += self.running_deliveries[:-1]

        new_orders = self.process.get_discrete_increase()
        print("Supply: ", self.current_supply)
        print("Orders: ", self.orders, " -> ", new_orders)
        print("Cost: ", self.cost)

        # add cost
        self.cost += self.orders * 1.0

        # place the end consumer order
        self.orders[-1] += new_orders
        print("End Order: ", self.cost)
        self.orders[:-1] += [self.agents[k].get_outgoing_orders(k, self.N, self.current_supply[k], self.orders[k + 1])
                             for k in range(self.N)]

        # update current supply
        self.running_deliveries[0] = self.orders[0]
        self.running_deliveries[1:] = np.minimum(self.orders[1:], self.current_supply)
        self.current_supply -= self.running_deliveries[1:]
        self.cost[:-1] += self.current_supply * 0.5

    def mult_steps(self, steps):
        """Performs multiple steps using the one step method."""

        [self.one_step() for _ in range(steps)]
