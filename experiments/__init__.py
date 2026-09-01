"""The studies themselves — each one an ``Experiment`` subclass or a notebook.

The simulator lives in deepracer-genesis; this package only configures and runs it.

    from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy

    class MyRun(Experiment):
        total_env_steps = 5_000_000
        group = "my_study"

        def pipeline(self):
            return FeatureEnvironment(num_envs=1024) >> VectorPolicy(keys=("state",))

Run it directly — ``MyRun().run()``, ``run(MyRun)``, a ``__main__`` block, or
``uv run python -m deepracer_genesis.experiment experiments.my_run:MyRun``. There is no
name registry: an experiment is referenced by its class. ``experiments/baselines/`` holds
the upstream examples worth subclassing.
"""
