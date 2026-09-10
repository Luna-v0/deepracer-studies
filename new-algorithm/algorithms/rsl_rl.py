"""rsl-rl-lib integration: build an OnPolicyRunner over a DeepRacerEnv.

The maintained training backend (the experiment API dispatches here via run_rsl).
"""

from __future__ import annotations

import copy
import os
import pickle


def build_runner(env, *, vision: bool, log_dir: str, device: str,
                 num_envs: int):
    """OnPolicyRunner over a DeepRacerEnv, with the cfgs pickled next to
    the logs (the eval CLI reloads them)."""
    from rsl_rl.runners import OnPolicyRunner

    from ..configs.cfgs import get_train_cfg

    train_cfg = get_train_cfg(vision=vision)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump({"env_cfg": env.cfg, "train_cfg": copy.deepcopy(train_cfg),
                     "num_envs": num_envs}, f)
    return OnPolicyRunner(env, train_cfg, log_dir, device=device)
