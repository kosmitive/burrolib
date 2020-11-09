# burro

[![Build Status](https://travis-ci.org/kosmitive/burrolib.png?branch=master)](https://travis-ci.org/kosmitive/burrolib?branch=develop)
[![Documentation Status](https://readthedocs.org/projects/burrolib/badge/?version=latest)](https://burrolib.readthedocs.io/en/latest/?badge=latest)

# Supply chain simulations / Beer game

This repository provides a small framework for supply chain simulations released by xploras. The framework is designed 
such that each participant in the chain can not see the full information but has access to previous and 
following nodes.

# Install necessary packages

You have to create an environment with the relevant libraries.

```
pip install numpy matplotlib
```

# Run the refill agent example

To start the agent simply run the following:
```
python examples/run_refill_agent.py
```

# Install burro as pip package
```
pip install -e .
```

## Contributors

[Markus Semmler](https://github.com/kosmitive/ )

[Thomas Lautenschläger](https://github.com/thlautenschlaeger/ )
