from abc import ABC, abstractmethod


class Agent(ABC):
    """Represents an interface for one agent. Different implementations
    can be used as this will be used inside the simulation to drive the
    different goods."""

    @abstractmethod
    def experience(self, state, action, reward, nxt_state, done):
        """
        Apply the agent to the environment and record the observations as experiences.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the agents from the new experiences.
        """
        pass

    @abstractmethod
    def sync(self):
        """
        Sync the agents,
            e.g. copy over weights from target network
            e.g. deleted old training data
        """
        pass

    @abstractmethod
    def act(self, state):
        """This method should calculate outgoing orders. Therefore several information is
        about itself is available.

        :param pos: The position in the product flow, where 0 means first node.
        :param clen: The length of the chain, for getting position in order flow.
        :param stock: The supply available in the stock.
        :param iorders: The deliveries requested by the node (pos + 1)
        :param oorders: The deliveries requested by you in the previous step.
        :param costs: The costs produced in last step.

        :return: How much should be ordered from (pos - 1)
        """
        pass
