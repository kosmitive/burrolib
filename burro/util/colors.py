def create_gradient(N, red, green, blue):
    """
    This method creates a gradient by using the supplied lists
    :param red: Red [from, to].
    :param green: Green [from, to].
    :param blue: Blue [from, to]
    :return: A list of [r,g,b] tuples.
    """

    # create color array
    def lin_func(x, lim):
        return (lim[0] + x * (lim[1] - lim[0])) / 255

    def colors(n):
        x = n / (N + 1)
        return lin_func(x, red), lin_func(x, green), lin_func(x, blue)

    return [(0.27, 0.5, 0.71)] + [colors(k) for k in range(N)] + [(0.8, 0.4, 0.4)]
