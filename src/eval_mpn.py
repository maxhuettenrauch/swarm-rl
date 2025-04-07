import supersuit as ss
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import SetObsWrapper, LimitedSetObsWrapper, PyGObsWrapper

from src.swarm_rl.utils import iterate

if __name__ == '__main__':

    env = RendezvousEnv(num_agents=4, render_mode='human')
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = PyGObsWrapper(env)
    env = ss.concat_vec_envs_v1(env, 1, 0, base_class='gymnasium')

    model = PPO.load("logs/best_model.zip")

    policy = model.policy

    ret = 0
    obs, info = env.reset(seed=0)
    for i in range(1000):
        actions, values, log_prob = policy(obs_as_tensor(obs, device='cpu'))
        obs, rews, dones, truncs, infos = env.step(actions.detach().numpy())
        print(actions[0])
        ret += rews[0]
    env.close()

    print(ret)