from typing import Any, Dict, List, Optional, Type, Union

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class MeanEmbeddingFeaturesExtractor(BaseFeaturesExtractor):
    """The mean embedding features extractor from the paper "Deep Reinforcement Learning for Swarm Systems"
    <https://jmlr.org/papers/v20/18-476.html>. It takes in a set of observations about neighboring agents and
    aggregates it into a latent representation to which a local observation is concatenated. The policy then
    further processes this representation."""
    def __init__(self, observation_space: spaces.Space, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim + observation_space.spaces['local_obs'].shape[0])
        self.embedding_dim = features_dim

        fc = torch.nn.Linear(observation_space.spaces['set_obs'].shape[1], self.embedding_dim)

        self.embedding_net = torch.nn.Sequential(fc, torch.nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        n_observed_agents = torch.sum(observations['set_obs'][:, :, -1], dim=-1, keepdim=True)

        embedded_obs = self.embedding_net(observations['set_obs'])

        mean_emb = torch.sum(embedded_obs, dim=-2) / torch.maximum(n_observed_agents, torch.tensor(1.0))

        # concat local obs
        mean_emb = torch.cat([mean_emb, observations['local_obs']], dim=-1)

        return mean_emb


class ActorCriticMeanEmbeddingPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = MeanEmbeddingFeaturesExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )