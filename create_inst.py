from init import *
from utils import *
import time

"""
Content Summary:
1. Create and save an instance of the UMVRP-L
   along with pre-computed value for shortest distances and paths
   from each node to another node in the given instance.
"""

rows, cols, A, numTasks, k, psi, centralized, visualizer, \
wall_prob, seed, collisions, exp_strat, _, _, _, _, _ = getParameters()
assert rows == cols
size = rows

out = init_valid_grid(rows, cols, A, numTasks, wall_prob=wall_prob, seed=seed, colis=collisions)

offlineTrainResult = offlineTrainCent(out["verts"], out["adjList"])

## save out 
save_instance(out, size, seed, offlineTrainResult)


