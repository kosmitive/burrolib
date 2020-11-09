from burrolib.agents.refill_agent import RefillAgent
from burrolib.games.beer_game import BeerGame
from burrolib.run import run


def test_run_refill_agent():
    n_players = 3
    game = BeerGame(n_players, 5.0)
    agent_factory = lambda i: RefillAgent(3, 8)
    run(game, agent_factory, n_players=n_players, n_steps=100, render=False)