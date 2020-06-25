from src.util.sampling import exp_sample


class DiscretePoissonProcess:
    """This class models a discrete poisson process. It uses
    the rate parameter and saves internally the state.
    """

    def __init__(self, lamb):
        """Initialize the process. Simply save the lambda internally."""
        self.lamb = lamb
        self.overhang_time = 0

    def get_discrete_increase(self, steps=1):
        """This method gets the number of events
        which occured between the timestep the method
        was last called, and the timestep lying steps
        in the future."""

        # precondition
        n = 0

        # loop body
        while True:

            # generate a exponential sample
            y = exp_sample(self.lamb)
            self.overhang_time += y

            # whenever we reached the limiting timestep
            # break
            if self.overhang_time > steps:
                self.overhang_time -= steps
                break

            # count up by one
            n += 1

        return n
