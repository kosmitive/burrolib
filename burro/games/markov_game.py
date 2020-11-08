class MarkovGame:

    def __init__(self, num_players):
        """

        :param num_players: Define the number of different players
        :param s0: The state transition function.
        """

        # save general settings
        self.num_players = num_players
        self.inner_state = self.reset()
        self.current_step = 0

    def step(self, actions):
        self.inner_state, rewards, done = self._state_transition(self.inner_state, actions)
        nxt_observations = self.get_observations()
        self.current_step += 1
        return nxt_observations, rewards, done

    def observation_dim(self, i):
        raise NotImplementedError

    def _state_transition(self, state, actions):
        raise NotImplementedError

    def _state_emission(self, state, i):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def reset(self):
        self.inner_state = self._reset()
        return self.get_observations()

    def get_observations(self):
        return [self._state_emission(self.inner_state, i) for i in range(self.num_players)]
