import math as m
import random as rnd


def unif_sample():
    """Generate a sample from U(0,1)."""
    return rnd.random()


def exp_sample(lamb):
    """This method generates a sample from an
    exponential distribution."""

    u = unif_sample()
    x = m.log(1 - u) / lamb
    return -x

def pp_disc_incr_sample(lamb):
    """Samples the increase for discrete timesteps t, where
    it basically counts the elements"""
