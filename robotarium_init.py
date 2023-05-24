import numpy as np
import argparse
import random

"""
The init file for robotarium based simulations is same as the discrete simulation's init file
however functions like bfsFindAgents is brought over from utils file and customized for 
continous space
"""
def bfsFindAgents(networkVertices, networkEdges, source, agentVertices):
    Q = []
    labels = {}
    prev = {}
    prev[str(source)] = None
    dist = -1

    for v in networkVertices:
        if v[0] == 0 and v[1] == 0:
            tup = (0.0, 0.0)
            labels[str(tup)] = False
        elif v[0] == 0:
            tup = (0.0, v[1])
            labels[str(tup)] = False
        elif v[1] == 0:
            tup = (v[0], 0.0)
            labels[str(tup)] = False
        else:
            labels[str(v)] = False

    Q.append(source)
    labels[str(source)] = True
    while(len(Q)) > 0:
        v = Q.pop(0)

        for edge in networkEdges:
            if edge[0] == v:
                if edge[1][0] == 0 and edge[1][1] == 0:
                    tup = (0.0, 0.0)
                elif edge[1][0] == 0:
                    tup = (0.0, edge[1][1])
                elif edge[1][1] == 0:
                    tup = (edge[1][0], 0.0)
                else:
                    tup = (edge[1][0], edge[1][1])
                if labels[str(tup)] == False:
                    labels[str(tup)] = True
                    prev[str(tup)] = v
                    Q.append(tup)
                    if tup in agentVertices:
                        return True

    return False

def init_valid_grid(numAgents, numTasks, wall_prob=0.2, seed=1234, colis=False):
    np.random.seed(seed)
    random.seed(seed)

    rows = 9
    cols = 9
    rstart_x = -0.8
    rstart_y = 0.8
    step = 0.2

    gridGraph = np.random.choice(np.arange(0,2), rows*cols, p=[wall_prob, 1-wall_prob])
    gridGraph = gridGraph.reshape((rows, cols))

    obs_dir = []

    ## populate vertices...
    vertices = []
    obstacles = []
    for i in range(rows):
        for j in range(cols):
            x_co = round(rstart_x + j*step, 1)
            y_co = round(rstart_y - i*step, 1)
            if gridGraph[i,j] == 1:
                vertices.append((x_co, y_co))
            else:
                obstacles.append((x_co, y_co))
    print("Grid Map::")
    print(gridGraph)

    ## populate edges...
    edgeList = []
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i,j] == 1:
                x_co = round(rstart_x + j*step, 1)
                y_co = round(rstart_y - i*step, 1)
                up = round(y_co+step, 1)
                down = round(y_co-step,1)
                left = round(x_co-step,1)
                right = round(x_co+step,1)
                if i-1 >= 0 and ((x_co,y_co),(x_co, up)) not in edgeList:
                    if gridGraph[i-1][j]==1:
                        edgeList.append(((x_co,y_co),(x_co, up)))
                        edgeList.append(((x_co, up),(x_co,y_co)))
                if j-1 >=0 and ((x_co,y_co),(left, y_co)) not in edgeList:
                    if gridGraph[i][j-1]==1:
                        edgeList.append(((x_co,y_co),(left, y_co)))
                        edgeList.append(((left, y_co),(x_co,y_co)))
                if i+1 < rows and ((x_co,y_co),(x_co, down)) not in edgeList:
                    if gridGraph[i+1][j]==1:
                        edgeList.append(((x_co,y_co),(x_co, down)))
                        edgeList.append(((x_co, down),(x_co,y_co)))
                if j+1 < cols and ((x_co,y_co),(right, y_co)) not in edgeList:
                    if gridGraph[i][j+1]==1:
                        edgeList.append(((x_co,y_co),(right, y_co)))
                        edgeList.append(((right, y_co),(x_co,y_co)))
                edgeList.append(((x_co,y_co),(x_co,y_co)))

    ## initialize agents...
    agentVertices = []
    for i in range(numAgents):
        done = False
        while (not done):
            agent_i = np.random.randint(0, high=rows)
            agent_j = np.random.randint(0, high=cols)

            ## check if cell is a free...
            if gridGraph[agent_i, agent_j] == 1:
                x_co = round(rstart_x + agent_j*step, 1)
                y_co = round(rstart_y - agent_i*step, 1)
                if colis == False:
                    if ((x_co, y_co) in agentVertices):
                        continue
                agentVertices.append((x_co, y_co))
                done = True

    free_vertices = list(set(vertices).difference(set(agentVertices)))
    taskVertices = [(-1,-1)]*numTasks
    for i in range(numTasks):
        done = False
        while not done:
            task = random.sample(free_vertices, 1)[0]
            ## check if task is reachable by some agent
            if bfsFindAgents(vertices, edgeList, task, agentVertices):
                done = True
                taskVertices[i] = task
                free_vertices.remove(task)

    return {"gridGraph":gridGraph, "adjList":edgeList, "verts":vertices, "agnt_verts":agentVertices, "task_verts":taskVertices, "obs_verts":obstacles, "obs_dir":obs_dir}


def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agt', required=False, default=7)
    parser.add_argument('--task', required=False, default=14)
    parser.add_argument('--k', required=False, default=3)
    parser.add_argument('--psi', required=False, default=2)
    parser.add_argument('--seed', required=False, default=34321)
    parser.add_argument('--exp_seed', required=False, default=1272922)
    parser.add_argument('--wall_pr', required=False, default=0.2)
    parser.add_argument('--only_base_pi', required=False, default=False, action='store_true')
    args = parser.parse_args()
    numAgents = (int)(args.agt)
    numTasks = (int)(args.task)
    k = (int)(args.k)
    psi = (int)(args.psi)
    seed = (int)(args.seed)
    exp_seed = (int)(args.exp_seed)
    wall_prob = (float)(args.wall_pr)
    only_base_policy = args.only_base_pi

    return numAgents, numTasks, k, psi, wall_prob, seed, only_base_policy, exp_seed
