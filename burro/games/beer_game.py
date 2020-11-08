import numpy as np
import matplotlib.pyplot as plt

from burro.processes.hawkes_process import HawkesProcess
from burro.games.markov_game import MarkovGame
from burro.processes.poisson_process import PoissonProcess
from burro.util.colors import create_gradient


class BeerGame(MarkovGame):

    def _state_emission(self, state, i):

        supply, orders, transported = state
        return np.stack([supply[i], transported[i], orders[i+1]])

    def observation_dim(self, i):
        return 3

    def _reset(self):

        return np.zeros(self.num_players), \
               np.zeros(self.num_players+1), \
               np.zeros(self.num_players+1)

    def _state_transition(self, state, actions):

        supply, orders, transported = state

        orders[-1] += self.consumer_process.next()
        orders[:-1] += actions

        transported[0] = orders[0]
        transported[1:] = np.minimum(orders[1:], supply)

        # subtract the number of
        supply += transported[:-1]
        orders -= transported

        cost = self.cost_storage * supply + self.cost_delay * orders[1:]
        supply -= transported[1:]

        # save cost and so on
        self.costs += cost
        self.generated_costs = cost
        self.generated_delay_costs = self.cost_delay * orders[1:]

        return (supply, orders, transported), -cost, False

    def render(self):

        if not self.rendered:

            # init the plot
            self.fig, self.ax = plt.subplots(figsize=(16, 8))
            plt.ion()
            plt.show()
            self.rendered = True

        plt.clf()
        ax = plt.axes()
        golden_ratio = self.gr
        golden_ratio_w = self.gr_w
        left = self.left
        bottom = self.bottom
        box_height = self.box_height
        box_width = self.box_width
        fs = self.fs
        N = self.num_players

        # add background
        v_grey = 0.85
        rect = plt.Rectangle((0, 0), 1, 0.3, facecolor=[v_grey] * 3)
        ax.add_patch(rect)
        rect = plt.Rectangle((0, 0.7), 1, 0.3, facecolor=[v_grey] * 3)
        ax.add_patch(rect)

        # get inner state
        supply, orders, transported = self.inner_state

        for k in range(N + 2):

            args = {}
            if 0 < k < N + 1:
                if self.generated_delay_costs[k - 1] > 0:
                    args['ec'] = 'darkred'

            rect = plt.Rectangle((left[k], bottom), box_width, box_height, facecolor=self.c_colors[k], **args)
            ax.add_patch(rect)

            if k < N + 1:
                arrow_product = plt.Arrow(left[k + 1] - golden_ratio_w,
                                          bottom + 2 * box_height / (2 + golden_ratio),
                                          golden_ratio_w, 0, color="#000000", width=0.01)
                ax.add_patch(arrow_product)

                arrow_order = plt.Arrow(left[k + 1], bottom + box_height / (2 + golden_ratio), -golden_ratio_w, 0,
                                        color="#555555", width=0.01)
                ax.add_patch(arrow_order)
                plt.text(left[k] + box_width + box_width * 0.5 * golden_ratio, bottom + 1.5 * box_height,
                         str(transported[k]),
                         fontsize=fs * golden_ratio, ha='center', va='bottom')

                plt.text(left[k] + box_width + box_width * 0.5 * golden_ratio, bottom - 0.5 * box_height,
                         str(orders[k]),
                         fontsize=fs * golden_ratio, ha='center', va='top')

            if k == 0:
                plt.text(left[k] + 0.5 * box_width, 0.9, "Gen:",
                         fontsize=fs * golden_ratio,
                         color='black', ha='center', va='center')
                plt.text(left[k] + 0.5 * box_width, 0.8, "Sum:", fontsize=fs * golden_ratio,
                         color='black', ha='center', va='center')
                plt.text(left[k] + 0.5 * box_width, bottom + 0.5 * box_height, "∞",
                         fontsize=fs, ha='center', va='center')

            if 0 < k < N + 1:
                plt.text(left[k] + 0.5 * box_width, bottom + 0.5 * box_height, str(supply[k - 1]),
                         fontsize=fs, ha='center', va='center')
                plt.text(left[k] + 0.5 * box_width, 0.9, str(self.generated_costs[k - 1]) + "€",
                         fontsize=fs * golden_ratio,
                         color='darkred', ha='center', va='center')
                plt.text(left[k] + 0.5 * box_width, 0.8, str(self.costs[k - 1]) + "€", fontsize=fs * golden_ratio,
                         color='darkred', ha='center', va='center')

        plt.title("SCM-SIM [" + str(self.current_step) + "]")
        plt.text(0.5, 0.15,
                 "Costs: " + str(np.sum(self.costs)) + "€",
                 fontsize=fs, color='darkred', ha='center', va='center')

        plt.axhline(y=0.7, linestyle='--', color='black')
        plt.axhline(y=0.3, linestyle='--', color='black')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.0001)

    plt.show()

    def __init__(self, chain_length, intensity, cost_storage=0.5, cost_delay=1.0, process='poisson'):

        super().__init__(chain_length)

        # create consumer process
        if process == 'poisson': self.consumer_process = PoissonProcess(intensity)
        elif process == 'hawkes': self.hawkes_process = HawkesProcess(intensity)

        self.cost_storage = cost_storage
        self.cost_delay = cost_delay
        self.last_cost = 0
        self.rendered = False
        self.costs = np.zeros(chain_length)

        # calc some drawing related things
        self.fig = None
        self.ax = None

        red = [111, 195]
        green = [137, 225]
        blue = [117, 169]
        self.c_colors = create_gradient(self.num_players, red, green, blue)

        self.gr = 1.64
        self.gr = 1 / self.gr
        h_marg_multiple = self.num_players

        self.width = self.num_players + (self.num_players + 3) * self.gr + 2
        self.height = 1 + 2 * h_marg_multiple * self.gr
        self.gr_w = self.gr / self.width
        self.gr_h = self.gr / self.height

        self.left = np.arange(self.num_players + 2) * (1 / self.width + self.gr_w) + self.gr_w
        self.bottom = self.num_players * self.gr_h
        self.box_width = 1 / self.width
        self.box_height = 1 / self.height
        self.fs = 245 * self.box_height
