import numpy as np
from sim.DiscretePoissonProcess import DiscretePoissonProcess
from sim.Simulator import Simulator
from sim.Agent import Agent
import matplotlib.collections as collections
import matplotlib.pyplot as plt

N = 10
sim = Simulator([Agent()] * N, N)
#sim.mult_steps(1000000)

golden_ratio = 1.64
golden_ratio = 1 / golden_ratio
h_marg_multiple = N

width = N + (N + 2) * golden_ratio + 1
height = 1 + 2 * h_marg_multiple * golden_ratio
golden_ratio_w = golden_ratio / width
golden_ratio_h = golden_ratio / height

left = np.arange(N + 1) * (1 / width + golden_ratio_w) + golden_ratio_w
bottom = N * golden_ratio_h
box_width = 1 / width
box_height = 1 / height

fig, ax = plt.subplots(figsize=(8, 4))
red = [111, 195]
green = [137, 225]
blue = [117, 169]

# create color array
lin_func = lambda x, lim: (lim[0] + x * (lim[1] - lim[0])) / 255
def colors(n):
    x = n / (N + 1)
    return (lin_func(x, red), lin_func(x, green), lin_func(x, blue))
cached_colors = [colors(k) for k in range(N)] + [(0.8, 0.4, 0.4)]

for k in range(N + 1):

    if k < N:
        arrow_product = plt.Arrow(left[k + 1] - golden_ratio_w, bottom + 2 * box_height / (2 + golden_ratio), golden_ratio_w, 0, color="#000000", width=0.01)
        ax.add_patch(arrow_product)

        arrow_order = plt.Arrow(left[k + 1], bottom + box_height / (2 + golden_ratio), -golden_ratio_w, 0,
                      color="#555555", width=0.01)
        ax.add_patch(arrow_order)
        plt.text(left[k] + 0.1 * box_width, bottom + 0.34 * box_height, "+05", fontsize=16.4)

    rect = plt.Rectangle((left[k], bottom), box_width, box_height, facecolor=cached_colors[k])
    ax.add_patch(rect)
    plt.text(left[k] + 0.1 * box_width, bottom + 0.34 * box_height, "+05", fontsize=16.4)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
#print(sim.cost)

