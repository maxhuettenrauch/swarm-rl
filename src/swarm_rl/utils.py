from gymnasium import spaces
from gymnasium.vector.utils import iterate


@iterate.register(spaces.Graph)
def _iterate_graph(space: spaces.Graph, items: spaces.GraphInstance):
    try:
        return [items]
    except TypeError as e:
        raise TypeError(
            f"Unable to iterate over the following elements: {items}"
        ) from e