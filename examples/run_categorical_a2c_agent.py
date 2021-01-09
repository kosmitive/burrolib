import sys
from copy import deepcopy

from burrolib.agents.categorical_agent import CategoricalAgent
from burrolib.games.beer_game import BeerGame
from burrolib.policies.categorical_policy import CategoricalPolicyModel
from burrolib.run import run

# sys.path.append(".")


# Initalize policy model
cat_policy = CategoricalPolicyModel(
    io_order_dim=3, max_order_size=5, p_hidden_size=(64, 64), vf_hidden_size=(64, 64)
)

cat_agent = CategoricalAgent(cat_policy)

# Setup simulator and agent
n_players = 4
game = BeerGame(n_players, intensity=5.0)
agent_factory = lambda i: deepcopy(cat_agent)

run(
    game=game,
    agent_factory=agent_factory,
    n_players=n_players,
    n_steps=100000,
    render=False,
)
