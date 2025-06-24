# Swarm RL: Graph Reinforcement Learning in Swarms

This repository contains a basic reimplementation of mean embeddings as published in "Deep Reinforcement Learning for Swarm Systems" (<https://jmlr.org/papers/v20/18-476.html>) in pytorch as well as pytorch geometric. Additionally, it contains a policy implementation based on Message Passing Networks as shown in Chapter 2 of my [dissertation](https://maxhuettenrauch.github.io/assets/pdf/diss_mh.pdf).

The tasks can be found in an extra [repository](https://github.com/maxhuettenrauch/swarm-zoo) and learning is based on a [fork](https://github.com/maxhuettenrauch/stable-baselines3/tree/feat/graph-spaces) of stable-baselines which has been extended to deal with graph observation spaces. Both will be installed as a dependency.

## Installation
This repo uses uv as a package manager. To install, first clone the repository and optionally install python 3.12
    
    uv python install 3.12

Create a virtual environment

    uv venv --python 3.12

and run

    uv sync

to install the dependencies.

## Usage
The folder `scripts` contains files for running training and evaluation. Adjust `log_path` folders to your needs. 