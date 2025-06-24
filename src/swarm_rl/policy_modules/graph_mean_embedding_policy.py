from typing import Any, Dict, List, Optional, Type, Union

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing


class GraphMeanEmbeddingModule(MessagePassing):
    """Mean embedding Module using graph terminology. The observations about other agents are modeled as edge features,
    while the local observations are modeled as node features."""
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, out_feature_dim: int = 64) -> None:
        super().__init__(aggr='mean', flow='target_to_source')

        self.msg_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_feature_dim, out_feature_dim),
            torch.nn.ReLU(),
        )

        self.upd_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_feature_dim + node_feature_dim, out_feature_dim),
            torch.nn.ReLU(),
        )

    def forward(self, node_features, edge_features, edge_index) -> Any:
        out = self.propagate(edge_index, x=node_features, edge_features=edge_features)
        return out

    def message(self, edge_features: Tensor) -> Tensor:
        out = self.msg_mlp(edge_features)
        return out

    def update(self, inputs: Tensor, x) -> Tensor:
        return self.upd_mlp(torch.cat([inputs, x], dim=-1))


class GraphMeanEmbeddingFeaturesExtractor(BaseFeaturesExtractor):
    """The mean embedding features extractor using graph terminology."""
    def __init__(self, observation_space: spaces.Graph, features_dim: int = 64, n_layers: int = 1) -> None:
        super().__init__(observation_space, features_dim)
        self.embedding_dim = features_dim
        self.n_layers = n_layers

        if n_layers > 1:
            self.lin_x = torch.nn.Linear(observation_space.node_space.shape[0], features_dim)
            self.lin_edge = torch.nn.Linear(observation_space.edge_space.shape[0], features_dim)

            self.mp_modules = nn.ModuleList([GraphMeanEmbeddingModule(features_dim, features_dim, features_dim) for _ in range(n_layers)])

        else:
            self.lin_x = torch.nn.Identity()
            self.lin_edge = torch.nn.Identity()
            self.mp_modules = nn.ModuleList([GraphMeanEmbeddingModule(edge_feature_dim=observation_space.edge_space.shape[0],
                                                                      node_feature_dim=observation_space.node_space.shape[0],
                                                                      out_feature_dim=features_dim)],
                                            )


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


class ActorCriticGraphMeanEmbeddingPolicy(ActorCriticPolicy):
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
            features_extractor_class: Type[BaseFeaturesExtractor] = GraphMeanEmbeddingFeaturesExtractor,
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