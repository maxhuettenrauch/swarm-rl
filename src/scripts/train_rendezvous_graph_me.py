import os
import time

import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from swarm_zoo.point_envs import RendezvousEnv
from swarm_zoo.point_envs.rendezvous import GraphObsWrapper

from src.swarm_rl.policy_modules.graph_mean_embedding_policy import ActorCriticGraphMeanEmbeddingPolicy
from src.swarm_rl.utils import iterate, GraphRolloutBuffer


def train():
    track = True
    time_stamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = "_".join(["rendezvous", time_stamp]) #  f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
    log_path = os.path.join("..", "logs", "rendezvous", time_stamp)
    if track:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e

        # tags = [*args.wandb_tags, f"v{sb3.__version__}"]
        run = wandb.init(
            project="swarm_rl",
            name=run_name,
            group="ppo_mpn",
            job_type="test",
            # entity=args.wandb_entity,
            # tags=tags,
            # config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        tensorboard_log = f"runs/{run_name}"

    n_agents = 10

    env = RendezvousEnv(num_agents=n_agents, render_mode=None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = GraphObsWrapper(env)
    env = ss.concat_vec_envs_v1(env, 4, 0, base_class='stable_baselines3')

    eval_env = RendezvousEnv(num_agents=n_agents, render_mode=None)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = GraphObsWrapper(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, 0, base_class='stable_baselines3')
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                 log_path=log_path, eval_freq=10000, deterministic=True, render=False,
                                 n_eval_episodes=1)

    checkpoint_callback = CheckpointCallback(save_freq=1024 * 16, save_path=log_path, )

    model = PPO(
        ActorCriticGraphMeanEmbeddingPolicy,
        env,
        rollout_buffer_class=GraphRolloutBuffer,
        rollout_buffer_kwargs={
            'n_agents': n_agents
        },
        tensorboard_log=tensorboard_log if track else None,
        policy_kwargs=dict(
            features_extractor_kwargs={'features_dim': 64,
                                       'n_hops':1
                                       },
            net_arch={'pi': [64], 'vf': [64]},
        ),
        verbose=3,
        batch_size=256,
        n_steps=2000,
    )

    model.learn(total_timesteps=100_000_000, callback=[eval_callback, checkpoint_callback])

    model.save(os.path.join(log_path, 'final_model'))

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


if __name__ == "__main__":
    train()