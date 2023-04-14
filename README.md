# Distributed-Multi-Agent-Rollout
Implementation of the improved DRL algorithm with provable sequential improvement wrt Greedy. 
## Parameters

1. `row`: (Int) Number of rows in grid world
2. `col`: (Int) Number of columns in grid world
3. `agt`: (Int) Number of agents in instance (randomly positioned)
4. `task`: (Int) Number of tasks in instance (randomly positioned)
5. `k`: (Int) individual agent view radius
6. `psi`: (Int) parameter to control cluster diameter
7. `wall_pr`: (float) probability of a specific grid cell being an obstacle during initialization
8. `seed`: (Int) random seed used to recreate instances
9. `no_colis`: (Bool) boolean flag; when used ensures that there is no agent co-location
10. `exp`: (Int) if 0 agents perform random walk based exploration, if greater than 0 agents perform Random Waypoint exploration with exploration distance given by value of exp
11. `only_base_pi`: (Bool) If true, only uses the greedy base heuristic within each cluster, otherwise performs standard multiagent rollout within each cluster
12. `cent`: (Bool) If true, executes the centralized version of the algorithm, otherwise the decentralized algorithm is executed
13. `vis`: (Bool) If true, executes the algorithm with the grid world visualized

## Usage

1. All instances will be generated and stored in a folder called `instances/` within the same directory where the `create_inst.py` file is located. 
2. Always initialize an instance using the `create_inst.py` file. It requires the same set of parameters as described above. 
3. `main.py` must be in the same directory as this `instances/` folder.
4. To execute the algorithm on an instance execute `main.py` with the same parameter set used while running `create_inst.py`. 

## Run Command Example

Create the instance of the simulation<br>
`python create_inst.py --row 10 --col 10 --agt 8 --task 25 --k 2 --psi 3 --seed 31223 --wall_pr 0.2 --no_colis --exp 4`<br>

Then run the simulation<br>
`python main.py --row 10 --col 10 --agt 8 --task 25 --k 2 --psi 3 --seed 31223 --wall_pr 0.2 --no_colis --exp 4 --vis`<br>