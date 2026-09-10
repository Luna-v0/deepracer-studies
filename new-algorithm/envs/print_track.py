"""Affiche la piste, puis fait rouler une voiture dedans (viewer Genesis).

La voiture est pilotee par CenterlineFollower, un simple correcteur P sur
l'ecart lateral et l'erreur de cap : aucun entrainement, aucun reseau. Le but
est seulement de voir une voiture bouger sur la piste.

    python new-algorithm/envs/print_track.py
"""

import os

import matplotlib.pyplot as plt
import torch

from deepracer_genesis import ASSETS_DIR
from deepracer_genesis.agents import CenterlineFollower
from deepracer_genesis.envs.track import TRACKS, load_route
from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.tools.track_builder import plot_track

TRACK = "reinvent_base"
NUM_ENVS = 4          # 4 voitures en parallele, lisible dans le viewer
STEPS = 3000          # duree de la balade
SHOW_TRACK_PLOT = False   # True = affiche d'abord le plan 2D (bloque jusqu'a fermeture)


def show_track(track: str = TRACK) -> None:
    """Trace le plan 2D de la piste avec matplotlib."""
    route = load_route(os.path.join(ASSETS_DIR, TRACKS[track][1]))
    plot_track(route)
    plt.show()


def drive(track: str = TRACK, num_envs: int = NUM_ENVS, steps: int = STEPS) -> None:
    """Ouvre le viewer et fait rouler des voitures scriptees sur la piste."""
    # meme chaine que les experiences : env -> politique. On s'arrete la, on
    # ne veut pas entrainer, juste construire le sim.
    spec = (
        FeatureEnvironment(num_envs=num_envs, tracks=(track,),
                           view="gui",            # ouvre le viewer interactif
                           realtime_factor=1.0)   # cadence temps reel
        >> VectorPolicy(keys=("state",))
    ).build()

    sim = Builder(spec).sim()
    sim.reset_idx(torch.arange(sim.num_envs, device=sim.device))

    driver = CenterlineFollower()
    for _ in range(steps):
        sim.step(driver.act(sim))   # (N, 2) = [braquage, acceleration]


if __name__ == "__main__":
    if SHOW_TRACK_PLOT:
        show_track()
    drive()
