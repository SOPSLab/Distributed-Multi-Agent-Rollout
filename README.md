# Distributed-Multi-Agent-Rollout
Implementation of the improved DRL algorithm with provable sequential improvement wrt Greedy. 

## Getting Started
Make sure `python>=3.9.12` has been installed using the instructions found [here](https://www.python.org/downloads/release/python-3100/)

And to make it easier to install all the dependencies please use `pip>=22.2.2` using the instructions found [here](https://pip.pypa.io/en/stable/installation/)

Install all the dependencies using:
```
pip install -r requirements.txt
```

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
14. `pure_greedy`: (Bool) If true, only uses the greedy policy for each agent without clustering, otherwise performs clustering with or without greedy base heuristic basis `only_base_pi` flag
15. `verbose`: (Int) prints out the algorithms execution information depending on the value
16. `depot`: (Bool) If true, executes the multiagent rollout algorithm with additional routing of all agents back to the original position of the leader agent
17. `run_num`: (Int) parameter used to set the random seed which controls the execution of the multiagent rollout algorithm with depots

## Basic Repository Structure 
### Discrete space simulation files
The main algorithm is housed in the `main.py` file with rest of the accompanying files used in the setup for the algorithm's execution.

The `init.py` script file is used to create and initialize random instances for the algorithms to run on. `create_inst.py` file contains the code to create an instance and a lookup table and save it as data files for repeated use. `utils.py` file contains the helper code functions and procedures used in other scripts.

### Continuous space simulation files
As before the main algorithm is housed in the `robotarium_main.py` file with `robotarium_main.py` used for running algorithm without visualization and `robotarium_main_remote.py` used to run algorithm on physical robots. The contents of the `utils.py` are split and moved to `robotarium_init.py` and the 2 main files to keep the codebase concise.
## Usage

### Simple execution flow for discrete space simulation
1. All instances will be generated and stored in a folder called `instances/` within the same directory where the `create_inst.py` file is located. 
2. Always initialize an instance using the `create_inst.py` file. It requires the same set of parameters as described above. 
3. `main.py` must be in the same directory as this `instances/` folder.
4. To execute the algorithm on an instance execute `main.py` with the same parameter set used while running `create_inst.py`.<br>

Create the instance of the simulation<br>
`python create_inst.py --row 10 --col 10 --agt 8 --task 25 --k 2 --psi 3 --seed 31223 --wall_pr 0.2 --no_colis --exp 4 --run_num 4`<br>

Then run the simulation<br>
`python main.py --row 10 --col 10 --agt 8 --task 25 --k 2 --psi 3 --seed 31223 --wall_pr 0.2 --no_colis --exp 4 --vis --run_num 4`<br>

### Simple execution flow for Physics based continuous space simulation
1. For continuous space simulations there is no need to specifically create an instance to run the simulation, to execute the algorithm simply specify parameters to the `robotarium_main.py` script, as given below.<br>

`python robotarium_main.py --k 4 --task 14 --seed 82931 --exp_seed 3729391`

To run the simulations on physical robots
1. Visit Robotarium platform's website and create an account as per the instructions given [here](https://www.robotarium.gatech.edu/get_started)
2. Once logged in and dashboard is accessible submit a request for a new experiment with `Estimated Duration` set to `900` seconds and `Number of Robots` set to `7`
3. Upload `robotarium_init.py` and `robotarium_main_remote.py` script files in the experiment files section and check `robotarium_main_remote.py` file to be main file
4. Submit the experiment

## Compute Information

Discrete space simulations of the algorithms were mostly run on Agave compute nodes and physics based continuous space simulations were run exclusively on Biodesign compute node.

### Agave compute nodes
Traditional x86 compute nodes were used which contain two Intel Xeon E5-2680 v4 CPUs running at 2.40GHz with at least 128 GB of RAM. There are 28 high speed Broadwell class CPU cores per node.

### Biodesign compute node
A single compute node with AMD Ryzen 9 7950X Processor with 16 cores and 32 threads along with 64 GB of RAM.

