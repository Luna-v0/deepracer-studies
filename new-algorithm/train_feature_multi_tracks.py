"""Sweep comparing TRPO against the built-in PPO across tracks and seeds.

Every combination of track, algorithm and seed is trained from scratch, then
summarized as mean completion rate per track. Headless by design: this is a
batch, not a watch session. Edit the three constants below to change what the
sweep covers; hard tracks need a larger TOTAL_ENV_STEPS to be learned at all.

    python new-algorithm/train_feature_multi_tracks.py
"""

from statistics import mean, pstdev

from algorithms.trpo import TRPO
from deepracer_genesis.experiment import (
    PPO,
    Algo,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)

TRACKS = ("reinvent_base", "Oval_track", "2022_reinvent_champ")
SEEDS = (0, 1, 2, 3, 4)
TOTAL_ENV_STEPS = 300_000


class MultiTrackRun(Experiment):
    """One run of the sweep: one algorithm, on one track, with one seed.

    Attributes:
        track: track the run trains on.
        use_trpo: train with TRPO instead of the built-in PPO.
        num_envs: cars simulated in parallel.
    """

    track = "reinvent_base"
    use_trpo = True
    num_envs = 64
    seed = 0
    total_env_steps = TOTAL_ENV_STEPS
    eval_every_steps = 0        # only the final eval matters for the comparison
    group = "multi_tracks"

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
                               tracks=(self.track,),
                               backend="gpu",
                               view="none")
            >> VectorPolicy(keys=("state",))
            >> self.algorithm()
            >> Evaluation(charts=False)
        )


def run_sweep() -> dict[tuple[str, str], list[float]]:
    """Trains every track x algorithm x seed combination.

    Returns:
        Completion rates keyed by (track, algorithm name).
    """
    scores: dict[tuple[str, str], list[float]] = {}
    for track in TRACKS:
        for use_trpo in (True, False):
            name = "TRPO" if use_trpo else "PPO"
            scores[(track, name)] = []
            for seed in SEEDS:
                record = MultiTrackRun(track=track, use_trpo=use_trpo, seed=seed,
                                       variant=f"{name.lower()}-{track}").run()
                value = record.metrics["completion_rate"]
                scores[(track, name)].append(value)
                print(f"[sweep] {track:22} {name:4} seed {seed}  "
                      f"completion {value:.3f}", flush=True)
    return scores


def print_summary(scores: dict[tuple[str, str], list[float]]) -> None:
    """Prints mean and spread of the completion rate per track and algorithm."""
    print(f"\n{'track':24} {'algo':5} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
    print("-" * 60)
    for track in TRACKS:
        for name in ("TRPO", "PPO"):
            values = scores.get((track, name))
            if not values:
                continue
            print(f"{track:24} {name:5} {mean(values):7.3f} {pstdev(values):7.3f} "
                  f"{min(values):7.3f} {max(values):7.3f}")


if __name__ == "__main__":
    print_summary(run_sweep())
