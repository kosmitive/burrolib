import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    """This class represents a simple simulator. It uses the defined
    agent interface to run the simulation."""

    def __init__(self, agents, process, N=4, plot=False):
        self.N = N
        self.outstanding = 0
        self.current_step = 0
        self.process = process
        self.plot = plot

        # define the data structure
        self.current_supply = 5 * np.ones([N], np.int32)
        self.orders = 3 * np.ones([N + 1], np.int32)
        self.running_deliveries = 3 * np.ones([N + 1], np.int32)
        self.costs = np.zeros([N])
        self.generated_costs = np.zeros([N])
        self.generated_delay_costs = np.zeros([N])

        # clone agents if necessary
        if not isinstance(agents, list):
            agents = [agents.clone() for _ in range(N)]

        # check length and save internally
        assert len(agents) == N
        self.agents = agents

        # init the plot
        if self.plot:
            self.fig, self.ax = plt.subplots(figsize=(16, 8))
            plt.ion()
            plt.show()

        # create colors
        red = [111, 195]
        green = [137, 225]
        blue = [117, 169]
        self.c_colors = self.create_gradient(red, green, blue)

        self.gr = 1.64
        self.gr = 1 / self.gr
        h_marg_multiple = N

        self.width = N + (N + 3) * self.gr + 2
        self.height = 1 + 2 * h_marg_multiple * self.gr
        self.gr_w = self.gr / self.width
        self.gr_h = self.gr / self.height

        self.left = np.arange(N + 2) * (1 / self.width + self.gr_w) + self.gr_w
        self.bottom = N * self.gr_h
        self.box_width = 1 / self.width
        self.box_height = 1 / self.height
        self.fs = 245 * self.box_height

        if self.plot:
            self.plot_frame()

    def create_gradient(self, red, green, blue):
        """
        This method creates a gradient by using the supplied lists

        :param red: Red [from, to].
        :param green: Green [from, to].
        :param blue: Blue [from, to]
        :return: A list of [r,g,b] tuples.
        """
        # create color array
        lin_func = lambda x, lim: (lim[0] + x * (lim[1] - lim[0])) / 255

        def colors(n):
            x = n / (self.N + 1)
            return lin_func(x, red), lin_func(x, green), lin_func(x, blue)

        return [(0.27, 0.5, 0.71)] + [colors(k) for k in range(self.N)] + [(0.8, 0.4, 0.4)]

    def one_step(self):
        """This method performs one discrete step in the simulator."""

        self.orders -= self.running_deliveries
        self.current_supply += self.running_deliveries[:-1]

        new_orders = self.process.get_discrete_increase()
        # print("Supply: ", self.current_supply)
        # print("Orders: ", self.orders, " -> ", new_orders)
        # print("Cost: ", self.costs)

        # add cost
        self.generated_delay_costs = self.orders[1:] * 1.0
        self.generated_costs = np.copy(self.generated_delay_costs)

        # place the end consumer order
        self.orders[-1] += new_orders
        # print("End Order: ", self.costs)
        self.orders[:-1] += [self.agents[k].get_outgoing_orders(k, self.N, self.current_supply[k],
                                                                self.orders[k + 1], self.orders[k], self.generated_costs[k])
                             for k in range(self.N)]

        # update current supply
        self.running_deliveries[0] = self.orders[0]
        self.running_deliveries[1:] = np.minimum(self.orders[1:], self.current_supply)
        self.current_supply -= self.running_deliveries[1:]
        self.generated_costs += self.current_supply * 0.5
        self.costs += self.generated_costs

        # plot
        self.current_step += 1
        if self.plot:
            self.plot_frame()
        return np.sum(self.generated_costs)

    def mult_steps(self, steps: int) -> np.ndarray:
        """Performs multiple steps using the one step method."""

        res = np.asarray([self.one_step() for _ in range(steps)])
        if self.plot:
            plt.ioff()
            plt.show(block=False)
        return res

    def plot_frame(self):

        plt.clf()
        ax = plt.axes()
        golden_ratio = self.gr
        golden_ratio_w = self.gr_w
        left = self.left
        bottom = self.bottom
        box_height = self.box_height
        box_width = self.box_width
        fs = self.fs
        N = self.N

        # add background
        v_grey = 0.85
        rect = plt.Rectangle((0, 0), 1, 0.3, facecolor=[v_grey] * 3)
        ax.add_patch(rect)
        rect = plt.Rectangle((0, 0.7), 1, 0.3, facecolor=[v_grey] * 3)
        ax.add_patch(rect)

        for k in range(N + 2):

            args = {}
            if 0 < k < N + 1:
                if self.generated_delay_costs[k - 1] > 0:
                    args['ec'] = 'darkred'

            rect = plt.Rectangle((left[k], bottom), box_width, box_height, facecolor=self.c_colors[k], **args)
            ax.add_patch(rect)

            if k < N + 1:
                arrow_product = plt.Arrow(left[k + 1] - golden_ratio_w, bottom + 2 * box_height / (2 + golden_ratio),
                                          golden_ratio_w, 0, color="#000000", width=0.01)
                ax.add_patch(arrow_product)

                arrow_order = plt.Arrow(left[k + 1], bottom + box_height / (2 + golden_ratio), -golden_ratio_w, 0,
                                        color="#555555", width=0.01)
                ax.add_patch(arrow_order)
                plt.text(left[k] + box_width + box_width * 0.5 * golden_ratio, bottom + 1.5 * box_height, str(self.running_deliveries[k]),
                         fontsize=fs * golden_ratio, ha='center', va='bottom')

                plt.text(left[k] + box_width + box_width * 0.5 * golden_ratio, bottom - 0.5 * box_height, str(self.orders[k]),
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
                plt.text(left[k] + 0.5 * box_width, bottom + 0.5 * box_height, str(self.current_supply[k - 1]),
                         fontsize=fs, ha='center', va='center')
                plt.text(left[k] + 0.5 * box_width, 0.9, str(self.generated_costs[k - 1]) + "€", fontsize=fs * golden_ratio,
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
