import torch
from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import PyGObsWrapper

from src.swarm_rl.policy_modules import GraphEmbedding

if __name__ == '__main__':

    env = RendezvousEnv(num_agents=4, render_mode=None)
    env = PyGObsWrapper(env)

    obs, info = env.reset(seed=0)

    mpn = GraphEmbedding(env.observation_space, 16)
    out = mpn(obs)

    print(out.numpy())
