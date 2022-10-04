from .custom_simple_spread.env import parallel_env as SimpleSpreadEnv
from .simple_cluster.env import parallel_env as SimpleClusterEnv

REGISTRY = {
    "custom_simple_spread": SimpleSpreadEnv,
    "simple_cluster": SimpleClusterEnv,
}
