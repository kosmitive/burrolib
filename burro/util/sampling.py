import torch
import torch.distributions as dist


def unif_sample(sample_shape=[1], low=0, high=1):
    """Generate a sample from U(low, high).

    :param sample_shape: The shape S
    :param low: Lower bound on uniform
    :param high: Upper bound on uniform
    :return: A matrix of shape S containg samples from the uniform.
    """
    d = dist.Uniform(low, high)
    return d.rsample(sample_shape)


def exp_sample(sample_shape=[1], lamb=1.0):
    """This method generates a sample from an exponential distribution.

    :param sample_shape: The shape S
    :param lamb: The rate of the exponential
    :return: A matrix of shape S containing samples from the exponential.
    """

    u = unif_sample(sample_shape)
    x = torch.log(1 - u) / lamb
    return -x


def gumbel_sample(sample_shape=[1], mu=0.0, beta=1.0):
    """Sample from the gumbel distribution by using the inverse transformation method.

    :param sample_shape: The shape S
    :param mu: The location of the gumbel.
    :param beta: The shape of the gumbel.
    :return: A matrix of shape S containg samples from the gumbel.
    """
    # check if params are valid
    assert beta > 0

    # create distribution and then sample from it

    uniform_samples = unif_sample(sample_shape)
    return mu - beta * torch.log(-torch.log(uniform_samples))


def categorical_sample(p, dim=-1):
    """Perform a categorical sample by using the gumbel max trick.
    See (https://arxiv.org/abs/1611.01144).

    :param p: The distribution probabilities.
    :param dim: The dimension over which the probabiites sum to one
    :return:The sampled actions.
    """

    z = gumbel_sample(p.size())
    return torch.argmax(p + z, dim=dim)
