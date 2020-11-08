import numpy as np

from burro.processes.discrete_point_process import DiscretePointProcess
from burro.util.type_checking import is_lambda


class HawkesProcess(DiscretePointProcess):

    """Types of hawkes processes."""
    HOMOGENEOUS = 1
    INHOMOGENEOUS = 2

    def __init__(self, intensity):
        """Creates a new hawkes process.

        :param intensity: Can be a float or a function f : R+ -> R+.
        """

        # encapsulate intensity in lambda
        if isinstance(intensity, float):
            self.type = HawkesProcess.HOMOGENEOUS
            self.intensity = lambda t: intensity

        elif is_lambda(intensity):
            self.type = HawkesProcess.INHOMOGENEOUS
            self.intensity = intensity

        self.arrivals = []
        self.eps = 1e-10
        self.t = 0

    def next(self):
        """Get number of arrivals for next timestep

        :return: The number of arrivals
        """

        t = self.t
        self.t += 1
        i = len(self.arrivals)

        while t < self.t:
            m = self.intensity(t + self.eps)
            e = np.exp(m)
            t = t + e
            u = np.random.uniform(0, m)

            if u < self.intensity(t):
                if t < self.t: i += 1
                self.arrivals.append(t)

        self.arrivals = self.arrivals[i:]
        print("hawkes generated ", i)
        return i
