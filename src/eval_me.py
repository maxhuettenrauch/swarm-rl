import supersuit as ss
import torch

from stable_baselines3 import PPO
from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import SetObsWrapper, LimitedSetObsWrapper

if __name__ == '__main__':

    env = RendezvousEnv(num_agents=8, render_mode='human')
    env = LimitedSetObsWrapper(env, max_observed_agents=2)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, 0, base_class='gymnasium')

    model = PPO.load("logs/rendezvous_20250306-125402.zip")

    policy = model.policy

    ret = 0
    obs, info = env.reset(seed=0)
    for i in range(1000):
        actions, values, log_prob = policy({key: torch.as_tensor(val, dtype=torch.float32) for key, val in obs.items()})
        obs, rews, dones, truncs, infos = env.step(actions.detach().numpy())
        print(actions[0])
        ret += rews[0]
    env.close()

    print(ret)