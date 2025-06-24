import numpy as np
import supersuit as ss

from stable_baselines3.common.utils import obs_as_tensor
from swarm_zoo.point_envs.rendezvous import RendezvousEnv, GraphObsWrapper

from src.swarm_rl.policy_modules.message_passing_network_policy import ActorCriticMessagePassingPolicy
from src.swarm_rl.utils import iterate


def get_rendezvous_env(num_agents=4, num_envs=1):
    env = RendezvousEnv(num_agents=num_agents, render_mode='human')
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = GraphObsWrapper(env)
    env = ss.concat_vec_envs_v1(env, num_envs, 0, base_class='gymnasium')
    return env


if __name__ == '__main__':

    env = get_rendezvous_env(num_envs=2)

    policy = ActorCriticMessagePassingPolicy(env.observation_space, env.action_space, lambda t: 0.01,
                                             features_extractor_kwargs={'features_dim': 8},
                                             net_arch={'pi': [8], 'vf': [8]})

    obs, info = env.reset(seed=0)

    ret = [0, 0]
    for i in range(1000):
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        actions, values, log_prob = policy(obs_as_tensor(obs, device='cpu'))
        obs, rews, dones, truncs, infos = env.step(np.clip(actions.detach().numpy(), env.action_space.low, env.action_space.high))
        if any(dones) or any(truncs):
            print('done or truncated, resetting env.')
            obs, info = env.reset()
        env.render()
        ret[0] += rews[0]
        # ret[1] += rews[4]
    env.close()

    print(ret)