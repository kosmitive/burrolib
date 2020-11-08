import numpy as np

def run(game, agent_factory, n_players, render=False):

    agents = [agent_factory(i) for i in range(n_players)]
    observations = game.reset()

    num_steps = 1000
    sync_steps = 20
    for k in range(num_steps):

        # obtain new action from each agent
        actions = np.asarray([agents[i].act(*list(observations[i])) for i in range(n_players)])
        nxt_observations, rewards, done = game.step(actions)
        if render: game.render()

        # add experience to each agent and train
        for i in range(n_players):
            agent = agents[i]
            agent.experience(observations[i], actions[i], rewards[i], nxt_observations[i], done)