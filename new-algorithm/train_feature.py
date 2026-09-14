"""Feature-vector training run, with the interactive viewer on by default.

Modeled on deepracer-genesis' examples/watch_live.py: hyperparameters are class
attributes, the stage chain lives in pipeline(), and .run() trains. The
use_trpo attribute selects the algorithm, so one script can compare both on the
same seeds and track.

    python new-algorithm/train_feature.py
"""

from algorithms.trpo import TRPO
from deepracer_genesis.experiment import (
    PPO,
    Algo,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)

HEADLESS = True    # no window: faster, follow along with `tensorboard --logdir runs/`


class TrainFeature(Experiment):
    """PPO or TRPO on the state vector, on reinvent_base.

    Attributes:
        num_envs: cars simulated in parallel; 64 stays readable in the viewer.
        total_env_steps: training budget, 1536 steps per iteration at 64 envs.
        eval_every_steps: interval between intermediate evaluations.
    """

    num_envs = 64
    seed = 0
    use_trpo = False
    total_env_steps = 300_000
    eval_every_steps = 100_000
    group = "new_algorithm"
    variant = "train_feature"

    def algorithm(self):
        """Returns the algorithm stage, TRPO or the built-in PPO."""
        if self.use_trpo:
            # lr only fits the critic here: the policy step size comes from the
            # trust region, and clip/epochs/minibatches are PPO-only knobs.
            return Algo(cls=TRPO, lr=1e-3, gamma=0.99, gae_lambda=0.95, horizon=24)
        return PPO(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                   epochs=5, minibatches=4, entropy_coef=0.01, horizon=24)

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=self.num_envs,
                               tracks=("reinvent_base",),
                               backend="gpu",     # Metal on macOS, CUDA elsewhere
                               view="none" if HEADLESS else "gui",
                               realtime_factor=1)
            >> VectorPolicy(keys=("state",))
            >> self.algorithm()
            >> Evaluation(charts=True)
        )


if __name__ == "__main__":
    for use_trpo in (True, False):
        for seed in range(5):
            record = TrainFeature(use_trpo=use_trpo, seed=seed).run()
            print("TRPO" if use_trpo else "PPO", "seed", seed,
                  "completion:", round(record.metrics["completion_rate"], 3))
