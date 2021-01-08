# burrolib

[![Build Status](https://travis-ci.org/kosmitive/burrolib.png?branch=master)](https://travis-ci.org/kosmitive/burrolib?branch=develop)
[![Documentation Status](https://readthedocs.org/projects/burrolib/badge/?version=latest)](https://burrolib.readthedocs.io/en/latest/?badge=latest)

This repository provides a small framework for multi-agent games released by xploras. The framework is designed 
such that each participant in the chain can not see the full information but has access to previous and 
following nodes.

# Install library & necessary packages

Prior to using the library execute

```
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

to create an environment and install the required packages. If you want to use rendering please install

```
sudo apt install python3-tk
```

the tk library. 

# Run the refill agent example

To start the agent simply run the following:
```
python examples/run_refill_agent.py
```

# Install burro as pip package
```
pip install -e .
```
