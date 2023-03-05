from utils import *

import numpy as np
import argparse
import sys
import random
from tqdm import tqdm

import time
# seed = np.random.randint(1000000)
# #print("Seed: ", seed)
# seed = 912985
# seed = 248839
# seed = 758250
# seed = 901620
# seed = 240768
# seed = 854885 # BestPos IndexError -- SOLVED
# seed = 147224 # BestPos IndexError
# seed = 433466 # --- Agents see task and yet do not solve it.
# seed = 127108 # --- Agents do not move
# seed = 519668 # --- Possibly too large 10 x 10 x 25 x 10 -- works now... Without the time.sleep(1), but a little slow.
# np.random.seed(seed)
# parser = argparse.ArgumentParser()
# parser.add_argument('row', type=int)
# parser.add_argument('col', type=int)
# parser.add_argument('agt', type=int)
# parser.add_argument('task', type=int)
# args = parser.parse_args()

# rows = args.row
# cols = args.col
# numAgents = args.agt
# numTasks = args.task

### Not needed...
# try:
# 	assert (numAgents >= numTasks)
# except AssertionError:
# 	#print("Number of agents must be greater than number of tasks")
# 	sys.exit()
###

def init_valid_grid(rows, cols, numAgents, numTasks, wall_prob=0.2, seed=1234, colis=True):
    print("Allowing Colissions? ", colis)
    np.random.seed(seed)
    random.seed(seed)

    gridGraph = np.random.choice(np.arange(0,2), rows*cols, p=[wall_prob, 1-wall_prob])
    gridGraph = gridGraph.reshape((rows, cols))

    ## populate vertices...
    vertices = []
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i,j] == 1:
                vertices.append((i,j))

    ## populate edges...
    edgeList = []
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i,j] == 1:
                if i-1 >= 0 and ((i,j),(i-1,j)) not in edgeList:
                    if gridGraph[i-1][j]==1:
                        edgeList.append(((i,j),(i-1,j)))
                        edgeList.append(((i-1,j),(i,j)))
                if j-1 >=0 and ((i,j),(i,j-1)) not in edgeList:
                    if gridGraph[i][j-1]==1:
                        edgeList.append(((i,j),(i,j-1)))
                        edgeList.append(((i,j-1),(i,j)))
                if i+1 < rows and ((i,j),(i+1,j)) not in edgeList:
                    if gridGraph[i+1][j]==1:
                        edgeList.append(((i,j),(i+1,j)))
                        edgeList.append(((i+1,j),(i,j)))
                if j+1 < cols and ((i,j),(i,j+1)) not in edgeList:
                    if gridGraph[i][j+1]==1:
                        edgeList.append(((i,j),(i,j+1)))
                        edgeList.append(((i,j+1),(i,j)))
                edgeList.append(((i,j),(i,j)))

    ## initialize agents...
    agentVertices = []
    for i in range(numAgents):
        done = False
        while (not done):
            agent_x = np.random.randint(0, high=rows)
            agent_y = np.random.randint(0, high=cols)

            ## check if cell is a free...
            if gridGraph[agent_x, agent_y] == 1:
                if colis == False:
                    if ((agent_x, agent_y) in agentVertices):
                        continue
                agentVertices.append((agent_x, agent_y))
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
    """
    ## get bfs trees...
    totalAgentsView = []
    for agent in tqdm(agentVertices):
        tree_verts, tree_edges = getBFSTree(vertices, edgeList, agent)
        #for vertex in tree_verts:
        #	totalAgentsView.append(vertex)
    ## remove duplicates from totalAgentsView
    totalAgentsView = list(set(tree_verts))
    ## remove agent vertices...
    totalAgentsView = list(set(totalAgentsView).difference(set(agentVertices)))

    ## sample tasks from this view...
    taskVertices = random.sample(totalAgentsView, numTasks)
    """

    print("Agents: ", agentVertices)
    print("Tasks: ", taskVertices)

    return {"gridGraph":gridGraph, "adjList":edgeList, "verts":vertices, "agnt_verts":agentVertices, "task_verts":taskVertices}



def get_valid_grid(rows, cols, numAgents, numTasks, wall_prob=0.2, seed=1234, colis=True):
    np.random.seed(seed)

    wall_prob = wall_prob
    vertices = []

    gridGraph = np.random.choice(np.arange(0,2), rows*cols, p=[wall_prob, 1-wall_prob])
    gridGraph = gridGraph.reshape((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i,j] == 1:
                vertices.append((i,j))
    #print(gridGraph)

    taskVertices = []
    # tasks
    # task_prob = 0.05
    # numTasks = (int)((1-wall_prob) * (row*col) * task_prob)

    for i in range(numTasks):
        done = False
        while (not done):
            task_x = np.random.randint(0, high=rows)
            task_y = np.random.randint(0, high=cols)

            ## check if cell already has a task in it...
            if ((task_x, task_y) in taskVertices):
                done = False
                continue

            ## check if vertex is a wall...
            if (gridGraph[task_x,task_y] == 1):
                taskVertices.append((task_x, task_y))
                done = True

    #print("Tasks: ", taskVertices)

    agentVertices = []
    for i in range(numAgents):
        done = False
        while (not done):
            agent_x = np.random.randint(0, high=rows)
            agent_y = np.random.randint(0, high=cols)
            if (gridGraph[agent_x, agent_y] == 1) and ((agent_x, agent_y) not in taskVertices):
                if (colis == False) and ((agent_x, agent_y) in agentVertices):
                    continue
                agentVertices.append((agent_x, agent_y))
                done = True

    #print("Agents: ", agentVertices)
    print("A")

    edgeList = []
    ## fill up adjacency list...
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i,j] == 1:
                if i-1 >= 0 and ((i,j),(i-1,j)) not in edgeList:
                    if gridGraph[i-1][j]==1:
                        edgeList.append(((i,j),(i-1,j)))
                        edgeList.append(((i-1,j),(i,j)))
                if j-1 >=0 and ((i,j),(i,j-1)) not in edgeList:
                    if gridGraph[i][j-1]==1:
                        edgeList.append(((i,j),(i,j-1)))
                        edgeList.append(((i,j-1),(i,j)))
                if i+1 < rows and ((i,j),(i+1,j)) not in edgeList:
                    if gridGraph[i+1][j]==1:
                        edgeList.append(((i,j),(i+1,j)))
                        edgeList.append(((i+1,j),(i,j)))
                if j+1 < cols and ((i,j),(i,j+1)) not in edgeList:
                    if gridGraph[i][j+1]==1:
                        edgeList.append(((i,j),(i,j+1)))
                        edgeList.append(((i,j+1),(i,j)))
                edgeList.append(((i,j),(i,j)))

    print("E")

    for task in taskVertices:
        print("dfs")
        task_path = dfs(vertices, edgeList, task)
        found = False
        for agent in agentVertices:
            if agent in task_path:
                # agent is found near this task!
                found = True
            else:
                found = False
                #print("No agents for task {}".format(task))
                return None

    return {"gridGraph":gridGraph, "adjList":edgeList, "verts":vertices, "agnt_verts":agentVertices, "task_verts":taskVertices}

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', required=True)
    parser.add_argument('--col', required=True)
    parser.add_argument('--agt', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--cent', required=False, default=False, action='store_true')
    parser.add_argument('--k', required=True)
    parser.add_argument('--psi', required=True)
    parser.add_argument('--vis', required=False, default=False, action='store_true')
    parser.add_argument('--seed', required=True)
    parser.add_argument('--wall_pr', required=True)
    parser.add_argument('--no_colis', required=False, default=True, action='store_false')
    parser.add_argument('--exp', required=True, type=int)
    parser.add_argument('--only_base_pi', required=False, default=False, action='store_true')
    # parser.add_argument('--seed', required=False, default=1234)
    args = parser.parse_args()

    rows = (int)(args.row)
    cols = (int)(args.col)
    numAgents = (int)(args.agt)
    numTasks = (int)(args.task)
    centralized = (args.cent)
    k = (int)(args.k)
    psi = (int)(args.psi)
    visualizer = (args.vis)
    seed = (int)(args.seed)
    wall_prob = (float)(args.wall_pr)
    collisions = (args.no_colis)
    exp_strat = args.exp
    only_base_policy = args.only_base_pi
    # seed = (int)(args.seed)

    assert(numTasks < rows*cols)
    assert(numAgents < rows*cols)

    # if centralized:
    #print("Centralized")
    # else:
    #print("Decentralized")

    return rows, cols, numAgents, numTasks, k, psi, centralized, visualizer, wall_prob, seed, collisions, exp_strat, only_base_policy





