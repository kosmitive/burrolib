import sys

sys.path.append(".")

from burro.agents.categorical_agent import CategoricalAgent
from burro.policies.categorical_policy import CategoricalPolicyModel
from burro.processes.discrete_poisson_process import DiscretePoissonProcess
from burro.sim.simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt

# initalize policy model
cat_policy = CategoricalPolicyModel(io_order_dim=3, max_order_size=20,
                                    p_hidden_size=(64, 64, 64),
                                    vf_hidden_size=(64, 64, 64))

# setup simulator and agent
N = 4
process = DiscretePoissonProcess(2.5)

plot_steps = 500

for i in range(10000):
    sim = Simulator(CategoricalAgent(cat_policy, gamma=0.99, batch_size=50), process=process, N=N)
    costs = sim.mult_steps(50)

    print(costs)

    # display a plot
    if i % plot_steps == 0:
        plt.figure()
        plt.title("Generated Costs")
        time = np.arange(len(costs))
        plt.plot(time, costs)
        plt.xlabel("t")
        plt.ylabel("costs")
        plt.show(block=True)
