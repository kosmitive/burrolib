from abc import ABC, abstractmethod


class BasePolicyModel(ABC):

    @abstractmethod
    def act(self, state):
        pass
