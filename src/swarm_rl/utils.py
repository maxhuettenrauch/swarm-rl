from typing import Optional, Union, NamedTuple, Generator

import numpy as np
import torch as th

from gymnasium import spaces
from gymnasium.spaces import GraphInstance
from gymnasium.vector.utils import iterate
from torch_geometric.data import Data, Batch

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecNormalize


@iterate.register(spaces.Graph)
def _iterate_graph(space: spaces.Graph, items: spaces.GraphInstance):
    try:
        return [items]
    except TypeError as e:
        raise TypeError(
            f"Unable to iterate over the following elements: {items}"
        ) from e


class GraphRolloutBufferSamples(NamedTuple):
    observations: Data | Batch
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class GraphRolloutBuffer(RolloutBuffer):

    observation_space: spaces.Graph
    obs_shape: tuple[GraphInstance]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_agents: int = 1
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.n_agents = n_agents
        self.n_envs = self.n_envs // self.n_agents
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs), dtype=GraphInstance)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.n_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step].flatten() + self.gamma * next_values.flatten() * next_non_terminal.flatten() - self.values[step].flatten()
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal.flatten() * last_gae_lam
            self.advantages[step] = last_gae_lam.reshape((self.n_envs, self.n_agents))
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: tuple[GraphInstance],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((-1, self.n_agents, self.action_dim))
        reward = reward.reshape((-1, self.n_agents))
        episode_start = episode_start.reshape((-1, self.n_agents))
        value = value.reshape((-1, self.n_agents))
        log_prob = log_prob.reshape((-1, self.n_agents))

        self.observations[self.pos] = obs
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> GraphRolloutBufferSamples:
        return GraphRolloutBufferSamples(
            observations=obs_as_tensor(tuple(ob[0] for ob in self.observations[batch_inds]), device='cpu'),
            actions=self.to_torch(self.actions[batch_inds].reshape(-1, self.action_dim)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )