"""
This experiment goal is to prove if a policy can race on a track using only the defined Feature vector
"""


from deepracer_genesis.experiment import (
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)


class FeatureVectorProof(Experiment):
    num_envs = 256
    seed = 0
    total_env_steps = 3_000_000
    eval_every_steps = 500_000
    variant = "feature_vector_proof"


    env = FeatureEnvironment(
        num_envs = num_envs,
        backend = "gpu",
        view = "gui",
        realtime_factor = 0,
        tracks = ('Singapore',
                  'Spain_track',
                  'Straight_track',
                  'Tokyo_Training_track',
                  'Vegas_track',
                  'reInvent2019_track',)
        ) 


    policy = VectorPolicy(keys=(("state",)))

    eval =  Evaluation(
            real_tracks = ("reinvent_base", "Oval_track"),
            eval_num_envs=32, charts=True
        )


    def pipeline(self):
        return self.env >> self.policy >> self.eval



if __name__ == '__main__':
    FeatureVectorProof().run()
