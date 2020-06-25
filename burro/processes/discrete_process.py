from abc import ABC, abstractmethod


class Process(ABC):
    """This class models a discrete poisson process. It uses
    the rate parameter and saves internally the state.
    """

    @abstractmethod
    def get_discrete_increase(self, steps=1):
        """This method gets the number of events
        which occured between the timestep the method
        was last called, and the timestep lying steps
        in the future."""
        pass
