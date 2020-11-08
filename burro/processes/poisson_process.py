from burro.processes.discrete_point_process import DiscretePointProcess
from burro.util.sampling import exp_sample


class PoissonProcess(DiscretePointProcess):
    """This class models a discrete poisson process. It uses
    the rate parameter and saves internally the state.
    """

    def __init__(self, lamb):
        """Initialize the process. Simply save the lambda internally."""
        self.lamb = lamb
        self.current_step = 0
        self.overhang_time = 0

    def next(self):
        """This method gets the number of events
        which occured between the timestep the method
        was last called, and the timestep lying steps
        in the future."""

        # precondition
        n = int(self.current_step > 0)

        # loop body
        while True:

            # generate a exponential sample
            y = exp_sample(lamb=self.lamb)
            self.overhang_time += y

            # whenever we reached the limiting timestep
            # break
            if self.overhang_time > 1:
                self.overhang_time -= 1
                break

            # count up by one
            n += 1

        print("poisson generated ", n)
        return n