import os
from deepracer_genesis import ASSETS_DIR
from deepracer_genesis.envs.track import TRACKS, load_route
from deepracer_genesis.tools.track_builder import plot_track
import matplotlib.pyplot as plt

route = load_route(os.path.join(ASSETS_DIR, TRACKS["reinvent_base"][1]))
plot_track(route)              # affiche dans un notebook
plt.show()
