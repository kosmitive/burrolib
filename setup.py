from distutils.core import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="burrolib",
    version="0.1.3",
    description="A multiagent toolkit",
    author="xploras",
    author_email="kosmitive@gmail.com",
    install_requires=required,
    packages=["burrolib"],
)
