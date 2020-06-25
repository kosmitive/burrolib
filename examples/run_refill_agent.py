import sys

sys.path.append(".")

from burro.agents.refill_agent import RefillAgent
from burro.processes.discrete_poisson_process import DiscretePoissonProcess
from burro.sim.simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt

# setup simulator and agent
N = 4
process = DiscretePoissonProcess(2.5)
sim = Simulator(RefillAgent(3, 8), process=process, N=N)
costs = sim.mult_steps(1000)

# display a plot
plt.figure()
plt.title("Generated Costs")
time = np.arange(len(costs))
plt.plot(time, costs)
plt.xlabel("t")
plt.ylabel("costs")
plt.show(block=True)
