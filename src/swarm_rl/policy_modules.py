from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch
import torch as th
import torch_geometric
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.typing import Adj


class MeanEmbedding(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Space, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim + observation_space.spaces['local_obs'].shape[0])
        self.embedding_dim = features_dim

        fc = torch.nn.Linear(observation_space.spaces['set_obs'].shape[1], self.embedding_dim)

        self.embedding_net = torch.nn.Sequential(fc, torch.nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Aggregates all embeddings in a mini-batch and computes the mean.

        Args:
            x (Tensor): The input node features.
            batch (LongTensor): Batch tensor mapping each node to its respective
                graph identifier.

        :rtype: :class:`Tensor`
        """

        # valid_obs = torch.where(observations['set_obs'][:, :, -1] == 1)
        n_observed_agents = torch.sum(observations['set_obs'][:, :, -1], dim=-1, keepdim=True)

        embedded_obs = self.embedding_net(observations['set_obs'])

        mean_emb = torch.sum(embedded_obs, dim=-2) / torch.maximum(n_observed_agents, torch.tensor(1.0))

        # concat local obs
        mean_emb = torch.cat([mean_emb, observations['local_obs']], dim=-1)

        return mean_emb

    def __repr__(self):
        return '{}(num_features={})'.format(self.__class__.__name__, self.embedding_dim)


class MeanEmbeddingPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = MeanEmbedding,
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


class MessagePassingModule(MessagePassing):
    def __init__(self, feature_dim: int = 16):
        super().__init__(aggr='sum', flow='target_to_source')

        self.msg_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dim, feature_dim),
            torch.nn.ReLU(),
        )

        self.upd_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dim, feature_dim),
            torch.nn.ReLU(),
        )

    # def message_and_aggregate(self, edge_index: Adj) -> Tensor:
    #     pass

    def edge_update(self) -> Tensor:
        pass

    def forward(self, node_features, edge_features, edge_index) -> Any:

        # node_features = self.lin_x(node_features)

        # alpha = self.edge_updater(edge_index, x=node_features,
        #                           edge_features=edge_features)

        out = self.propagate(edge_index, x=node_features, edge_features=edge_features)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_features: Tensor) -> Tensor:
        out = self.msg_mlp(torch.cat([x_j, edge_features], dim=-1))
        return out

    def update(self, inputs: Tensor, x) -> Tensor:
        return self.upd_mlp(torch.cat([inputs, x], dim=-1))


# TODO: Message passing feature extractor based on GATv2Conv



class GraphEmbedding(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Graph, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim)
        self.embedding_dim = features_dim

        self.lin_x = torch.nn.Linear(observation_space.node_space.shape[0], features_dim)
        self.lin_edge = torch.nn.Linear(observation_space.edge_space.shape[0], features_dim)

        self.mpn1 = MessagePassingModule(features_dim)
        self.mpn2 = MessagePassingModule(features_dim)

        fc = torch.nn.Linear(1, self.embedding_dim)

        self.embedding_net = torch.nn.Sequential(fc, torch.nn.ReLU())

    def forward(self, observations: Dict[str, spaces.GraphInstance]) -> torch.Tensor:
        # x = torch.as_tensor(observations['agent_0']['graph'].nodes, dtype=torch.float32)
        # edge_attr = torch.as_tensor(observations['agent_0']['graph'].edges, dtype=torch.float32)
        # edge_index = torch.as_tensor(observations['agent_0']['graph'].edge_links, dtype=torch.int64).t().contiguous()

        x = observations.x
        edge_attr = observations.edge_attr
        edge_index = observations.edge_index

        # embed the graph into a latent graph embedding
        x = self.lin_x(x)
        edge_attr = self.lin_edge(edge_attr)

        x = self.mpn1(x, edge_attr, edge_index)
        x = self.mpn2(x, edge_attr, edge_index)

        return x


class GraphEmbeddingPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = GraphEmbedding,
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

    # def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     graph: spaces.GraphInstance = obs['graph']
    #
    #     edge_index = torch.tensor(graph.edge_links).t().contiguous()
    #     edge_attr = torch.tensor(graph.edges)
    #     node_attr = torch.tensor(graph.nodes)
    #
    #     agent_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    #     return super().forward(agent_graph, deterministic)