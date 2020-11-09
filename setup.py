from distutils.core import setup

import numpy as np


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='burrolib',
      version='0.1.2',
      description='burro - a supply chain simulator',
      author='Markus Semmler, Thomas Lautenschlaeger',
      author_email='dev@xploras.net',
      install_requires=required,
      packages=['burrolib'],
      include_dirs=[np.get_include()],)
