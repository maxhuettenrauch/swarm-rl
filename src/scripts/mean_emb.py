import supersuit as ss
import torch
from swarm_zoo.point_envs.rendezvous import RendezvousEnv, SetObsWrapper

from src.swarm_rl.policy_modules import MeanEmbeddingPolicy


if __name__ == '__main__':

    env = RendezvousEnv(num_agents=4, render_mode=None)
    env = SetObsWrapper(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, 0, base_class='gymnasium')
    policy = MeanEmbeddingPolicy(env.observation_space, env.action_space, lambda t: 0.01,
                                 features_extractor_kwargs={'features_dim': 64},
                                 net_arch={'pi': [64], 'vf': [64]})
    obs, info = env.reset(seed=0)

    ret = [0, 0]
    for i in range(1000):
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        actions, values, log_prob = policy({key: torch.as_tensor(val, dtype=torch.float32) for key, val in obs.items()})
        obs, rews, dones, truncs, infos = env.step(actions.detach().numpy())
        ret[0] += rews[0]
        ret[1] += rews[4]
    env.close()

    print(ret)