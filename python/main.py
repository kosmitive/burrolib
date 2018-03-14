import numpy as np
from python.sim.DiscretePoissonProcess import DiscretePoissonProcess
from python.sim.Simulator import Simulator
from python.sim.Agent import Agent
import matplotlib.collections as collections
import matplotlib.pyplot as plt

N = 8
sim = Simulator([Agent()] * N, N)
sim.mult_steps(100000)