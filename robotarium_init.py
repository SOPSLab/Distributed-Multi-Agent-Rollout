import numpy as np
import argparse
import random

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
    print("Allowing Colissions? ", colis)
    np.random.seed(seed)
    random.seed(seed)

    rows = 9
    cols = 9
    rstart_x = -0.8
    rstart_y = 0.8
    step = 0.2

    gridGraph = np.random.choice(np.arange(0,2), rows*cols, p=[wall_prob, 1-wall_prob])
    gridGraph = gridGraph.reshape((rows, cols))

    ## Uncomment for static obstacles with better graphics in robotarium
    # gridGraph = np.ones((rows, cols), dtype=int)

    # obs = [(1,1),(1,3),(1,4),(1,5),(1,7),(2,4),(3,1),(3,3),(3,4),(3,5),(3,7),(5,1),(5,2),(5,3),(5,5),(5,6),(5,7),(6,3),(6,5),(7,1),(7,3),(7,5),(7,7)]
    # obs_dir = [2,0,0,0,2,1,2,0,0,0,2,0,0,0,0,0,0,1,1,2,1,1,2]
    obs_dir = []
    # agents = [(2,2),(2,6),(4,1),(4,4),(4,7),(6,1),(6,7),(7,4)]
    # agents = [(2,2),(2,6),(4,1),(7,4),(4,7),(6,1),(6,7)]

	# print("Grid Map::")
    # print(gridGraph)
    
    ## populate vertices...
    vertices = []
    obstacles = []
    for i in range(rows):
        for j in range(cols):
            # if (i,j) in obs:
            #     gridGraph[i][j] = 0
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
            # agent_i = agents[i][0]
            # agent_j = agents[i][1]

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

    return {"gridGraph":gridGraph, "adjList":edgeList, "verts":vertices, "agnt_verts":agentVertices, "task_verts":taskVertices, "obs_verts":obstacles, "obs_dir":obs_dir}


def getParameters():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--row', required=True)
	# parser.add_argument('--col', required=True)
    parser.add_argument('--agt', required=False, default=7)
    parser.add_argument('--task', required=False, default=14)
    # parser.add_argument('--cent', required=False, default=False, action='store_true')
    parser.add_argument('--k', required=False, default=3)
    parser.add_argument('--psi', required=False, default=2)
    # parser.add_argument('--vis', required=False, default=False, action='store_true')
    #current 10
    #24921
    #34321
    #82931
    #56122
    #73192
    #10932
    #69347
    #48369
    #99918
    #66182

    #18621
    #17392
    #29038
    #36148
    #47771
    #52918
    #62110
    #73224
    #82710
    #91883
    parser.add_argument('--seed', required=False, default=34321)
    parser.add_argument('--exp_seed', required=False, default=1272922)
    parser.add_argument('--wall_pr', required=False, default=0.2)
    # parser.add_argument('--no_colis', required=False, default=True, action='store_false')
    # parser.add_argument('--exp', required=True, type=int)
    parser.add_argument('--only_base_pi', required=False, default=False, action='store_true')
    # parser.add_argument('--seed', required=False, default=1234)
    args = parser.parse_args()  
    # rows = (int)(args.row)
    # cols = (int)(args.col)
    numAgents = (int)(args.agt)
    numTasks = (int)(args.task)
    # centralized = (args.cent)
    k = (int)(args.k)
    psi = (int)(args.psi)
    # visualizer = (args.vis)
    seed = (int)(args.seed)
    exp_seed = (int)(args.exp_seed)
    wall_prob = (float)(args.wall_pr)
    # collisions = (args.no_colis)
    # exp_strat = args.exp
    only_base_policy = args.only_base_pi
    # seed = (int)(args.seed)   
    # assert(numTasks < rows*cols)
    # assert(numAgents < rows*cols) 
    # if centralized:
    	#print("Centralized")
    # else:
    	#print("Decentralized") 
    return numAgents, numTasks, k, psi, wall_prob, seed, only_base_policy, exp_seed





# init_valid_grid(7, 14, 0.2, 56122, False)