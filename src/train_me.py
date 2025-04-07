import time
from tokenize import group

import supersuit as ss
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import SetObsWrapper, LimitedSetObsWrapper

from src.swarm_rl.policy_modules import MeanEmbeddingPolicy


def train():
    track = True
    if track:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        run_name = "test_swarm_sb3" #  f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        # tags = [*args.wandb_tags, f"v{sb3.__version__}"]
        run = wandb.init(
            project="swarm_rl",
            name=run_name,
            group="ppo",
            job_type="test",
            # entity=args.wandb_entity,
            # tags=tags,
            # config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        tensorboard_log = f"runs/{run_name}"

    env = RendezvousEnv(num_agents=8, render_mode=None)
    env = SetObsWrapper(env)
    # env = LimitedSetObsWrapper(env, max_observed_agents=2)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, 0, base_class='stable_baselines3')
    # policy = MeanEmbeddingPolicy(env.observation_space, env.action_space, lambda t: 0.01,
    #                              features_extractor_kwargs={'features_dim': 64},
    #                              net_arch={'pi': [64], 'vf': [64]})

    eval_env = RendezvousEnv(num_agents=8, render_mode=None)
    eval_env = SetObsWrapper(eval_env)
    # eval_env = LimitedSetObsWrapper(eval_env, max_observed_agents=2)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, 0, base_class='stable_baselines3')
    eval_callback = EvalCallback(eval_env, best_model_save_path='logs/',
                                 log_path='logs/', eval_freq=10000, deterministic=True, render=False,
                                 n_eval_episodes=1)

    model = PPO(
        MeanEmbeddingPolicy,
        env,
        tensorboard_log=tensorboard_log if track else None,
        policy_kwargs=dict(
            features_extractor_kwargs={'features_dim': 64},
            net_arch={'pi': [64], 'vf': [64]},
        ),
        verbose=3,
        batch_size=256,
        n_steps=4000,
    )

    model.learn(total_timesteps=10_000_000, callback=eval_callback)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


if __name__ == "__main__":
    train()