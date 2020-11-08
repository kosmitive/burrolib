import numpy as np

def run(game, agent_factory, n_players, n_steps, render=False, n_sync_steps=10):

    agents = [agent_factory(i) for i in range(n_players)]
    observations = game.reset()

    # repeat
    for k in range(n_steps):

        # obtain new action from each agent
        actions = np.asarray([agents[i].act(observations[i]) for i in range(n_players)])
        nxt_observations, rewards, done = game.step(actions)
        if render: game.render()

        # add experience to each agent and train
        for i in range(n_players):
            agent = agents[i]
            agent.experience(observations[i], actions[i], rewards[i], nxt_observations[i], done)

            # train agent
            agent.train()
            if k % n_sync_steps:
                agent.sync()