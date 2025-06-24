from typing import Any, Dict, List, Optional, Type, Union

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing


class MessagePassingModule(MessagePassing):
    def __init__(self, feature_dim: int = 16):
        super().__init__(aggr='mean', flow='target_to_source')

        self.msg_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * feature_dim, feature_dim),
            torch.nn.ReLU(),
        )

        self.upd_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dim, feature_dim),
            torch.nn.ReLU(),
        )

    def forward(self, node_features, edge_features, edge_index) -> Any:
        out = self.propagate(edge_index, x=node_features, edge_features=edge_features)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_features: Tensor) -> Tensor:
        out = self.msg_mlp(torch.cat([x_i, x_j, edge_features], dim=-1))
        return out

    def update(self, inputs: Tensor, x) -> Tensor:
        return self.upd_mlp(torch.cat([inputs, x], dim=-1))


class MessagePassingNetworkFeaturesExtractor(BaseFeaturesExtractor):
    """A features extractor that uses message passing. In addition to embedding the edge features, it also takes in
    node features of neighboring agents. Through multiple hops, information can be passed from agents that are not
    observed directly."""
    def __init__(self, observation_space: spaces.Graph, features_dim: int = 0, n_hops: int = 2) -> None:
        super().__init__(observation_space, features_dim)
        self.embedding_dim = features_dim

        self.lin_x = torch.nn.Linear(observation_space.node_space.shape[0], features_dim)
        self.lin_edge = torch.nn.Linear(observation_space.edge_space.shape[0], features_dim)

        self.mp_modules = nn.ModuleList([MessagePassingModule(features_dim) for _ in range(n_hops)])


    def forward(self, observations: Dict[str, spaces.GraphInstance]) -> torch.Tensor:
        x = observations.x
        edge_attr = observations.edge_attr
        edge_index = observations.edge_index

        # embed the graph into a latent graph embedding
        x = self.lin_x(x)
        edge_attr = self.lin_edge(edge_attr)

        for mpn in self.mp_modules:
            x = mpn(x, edge_attr, edge_index)

        return x


class ActorCriticMessagePassingPolicy(ActorCriticPolicy):
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
            features_extractor_class: Type[BaseFeaturesExtractor] = MessagePassingNetworkFeaturesExtractor,
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
