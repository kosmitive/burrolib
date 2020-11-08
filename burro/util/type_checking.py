def is_lambda(v):
    """Check if the passed variable is a lambda"""
    l = lambda: 0
    return isinstance(v, type(l)) and v.__name__ == l.__name__
