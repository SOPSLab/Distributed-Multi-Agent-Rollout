import os
import pickle
from tqdm import tqdm

"""
Utils file
Content Summary:
1. Used by other scripts of the repository
2. Functions to load and store the instances to and from a data file
3. 
"""

def save_file(filename, data):
    with open (filename, "wb") as fp:
        pickle.dump(data, fp)

def save_dict(filename, data):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filename):
    with open(filename, 'rb') as fp:
        file = pickle.load(fp)

    return file

def load_file(filename):
    with open(filename, "rb") as fp:
        file = pickle.load(fp)

    return file

"""
An instance object is streamed and saved as a data file which can be used repeatedly
"""
def save_instance(instance, size, seed, offlineTraining):
    directory = "./instances/"+str(size)+"x"+str(size)+"/"+str(seed)

    if os.path.exists(directory) == False:
        os.makedirs(directory)

    ## start saving instance...
    gridGraph = instance['gridGraph'].tolist()
    adjList = instance['adjList']
    vertices = instance['verts']
    agentVertices = instance['agnt_verts']
    taskVertices = instance['task_verts']

    save_file(directory + "/offlineTrain", offlineTraining)

    save_file(directory + "/gridGraph", gridGraph)
    save_file(directory + "/adjList", adjList)
    save_file(directory + "/vertices", vertices)
    save_file(directory + "/agentVertices", agentVertices)
    save_file(directory + "/taskVertices", taskVertices)

"""
An instance object is read from data file with a specific seed and size as instance identification
"""
def load_instance(size, seed):
    path = "./instances/"+str(size)+"x"+str(size)+"/"+str(seed)
    print(path)
    assert os.path.exists(path) == True

    gridGraph = load_file(path + "/gridGraph")
    edgeList = load_file(path + "/adjList")
    vertices = load_file(path + "/vertices")
    agentVertices = load_file(path + "/agentVertices")
    taskVertices = load_file(path + "/taskVertices")

    offlineTraining = load_dict(path + "/offlineTrain")

    return {"gridGraph":gridGraph, "adjList":edgeList, "verts":vertices, "agnt_verts":agentVertices, "task_verts":taskVertices}, offlineTraining


"""
Given a graph structure creates a cache of all the paths between any two nodes in the graph
An iterative implementation of the breath first search is used to find the shortest path
between any two given nodes
"""
def offlineTrainCent(networkVertices, networkEdges):
    bfsTrees = {}

    for v in tqdm(networkVertices):
        tree_vertices, tree_edges = getBFSTree(networkVertices, networkEdges, v)
        bfsTrees[str(v)] = (tree_vertices, tree_edges)

    lookupDict = {}
    for v in tqdm(networkVertices):
        lookupDict[str(v)] = {}
        for u in networkVertices:
            if u != v:
                try:
                    dist, path = bfsLookupSP(bfsTrees[str(v)], u)
                    lookupDict[str(v)][str(u)] = (dist, path[1])
                except AssertionError:
                    continue

    return lookupDict


"""
Iterative implementation of a breadth first search on a graph structure
which return if the goal node is found or not, (starting from a given root node)
"""
def bfs(vertices,edges,root,goal):
    Q=[]
    labels={}
    for v in vertices:
        labels[str(v)]=False
    Q.append(root)
    labels[str(root)]=True
    while (len(Q))>0:
        v=Q.pop(0)
        if v==goal:
            return True
        for e in edges:
            if e[0]==v:
                if labels[str(e[1])]==False:
                    labels[str(e[1])]=True
                    Q.append(e[1])
    return False

"""
Iterative implementation of breath first traversal of the graph sructure while
simultaneously building a tree structure as the breath first traversal progresses
"""
def getBFSTree(vertices, edges, source):
    tree_vertices = []
    tree_edges = []

    Q = []
    labels = {}
    prev = {}

    for v in vertices:
        labels[str(v)] = False

    Q.append(source)
    labels[str(source)] = True
    while(len(Q)) > 0:
        v = Q.pop(0)
        tree_vertices.append(v)
        if v == source:
            tree_edges.append((v, None))

        for edge in edges:
            if edge[0] == v:
                if labels[str(edge[1])] == False:
                    prev[str(edge[1])] = v
                    Q.append(edge[1])
                    tree_edges.append((edge[1], edge[0]))
                    labels[str(edge[1])] = True

    return tree_vertices, tree_edges

"""
An iterative implementation of the breath first search, used to find the shortest path
to a given target node in a tree structure starting with the root of the tree 
"""
def bfsLookupSP(bfsTree, target):
    tree_vertices = bfsTree[0]
    tree_edges = bfsTree[1]

    dist = 0
    path = []

    assert target in tree_vertices
    path.append(target)

    for edge in tree_edges:
        if target == edge[0]:
            curr_node = edge[0]
            break

    while (curr_node != None):
        for edge in tree_edges:
            if curr_node == edge[0]:
                curr_node = edge[1]

                if curr_node != None:
                    path.append(curr_node)
                    dist += 1

                break

    return dist, list(reversed(path))

## Not used
def bfShortestPath(networkVertices, networkEdges, source, target):
    Q = []
    labels = {}
    prev = {}
    prev[str(source)] = None
    dist = -1

    for v in networkVertices:
        labels[str(v)] = False

    Q.append(source)
    labels[str(source)] = True
    while(len(Q)) > 0:
        v = Q.pop(0)

        if v == target:
            ## find path
            S = []
            t = target
            if prev[str(t)] != None or t==source:
                while t != None:
                    S.append(t)
                    t=prev[str(t)]
                    dist += 1
            return dist, list(reversed(S))
        ## not target node
        for edge in networkEdges:
            if edge[0] == v:
                if labels[str(edge[1])] == False:
                    labels[str(edge[1])] = True
                    prev[str(edge[1])] = v
                    Q.append(edge[1])

"""
Iterative breath first traversal of a given graph structure to 
check if any one of the agent nodes are reachable from the source node
"""
def bfsFindAgents(networkVertices, networkEdges, source, agentVertices):
    Q = []
    labels = {}
    prev = {}
    prev[str(source)] = None
    dist = -1

    for v in networkVertices:
        labels[str(v)] = False

    Q.append(source)
    labels[str(source)] = True
    while(len(Q)) > 0:
        v = Q.pop(0)

        for edge in networkEdges:
            if edge[0] == v:
                if labels[str(edge[1])] == False:
                    labels[str(edge[1])] = True
                    prev[str(edge[1])] = v
                    Q.append(edge[1])
                    if edge[1] in agentVertices:
                        return True

    return False

"""
Uses breath first traversal to find the path and distance between the given source node 
and the closest task node from a list of task nodes
"""
def bfsNearestTask(networkVertices, networkEdges, source, taskVertices):
    Q = []
    labels = {}
    prev = {}
    prev[str(source)] = None
    dist = -1

    for v in networkVertices:
        labels[str(v)] = False

    Q.append(source)
    labels[str(source)] = True
    while(len(Q)) > 0:
        v = Q.pop(0)

        for edge in networkEdges:
            if edge[0] == v:
                if labels[str(edge[1])] == False:
                    labels[str(edge[1])] = True
                    prev[str(edge[1])] = v
                    Q.append(edge[1])
                    if edge[1] in taskVertices:
                        t = edge[1]
                        if prev[str(t)] != None or t==source:
                            path = []
                            while t != None:
                                path.append(t)
                                t = prev[str(t)]
                                dist += 1
                        return dist, list(reversed(path))

    return None,None

"""
Iterative breath first traversal of a given graph structure to find the shortest path
between given two nodes
"""
def dirShortestPath(networkVertices,networkEdges,source,target):
    Q=[]
    dist={}
    prev={}

    assert target in networkVertices
    assert source in networkVertices

    for v in networkVertices:
        dist[str(v)]= 9999999999
        prev[str(v)]=None
        Q.append(v)
    dist[str(source)]=0
    while len(Q)>0:
        uNum=9999999999
        u=None
        for q in Q:
            if dist[str(q)]<=uNum:
                u=q
                uNum=dist[str(q)]
        Q.remove(u)
        if u == target:
            S=[]
            t=target
            if prev[str(t)] != None or t==source:
                while t != None:
                    S.append(t)
                    t=prev[str(t)]
                return dist[str(target)],list(reversed(S))
        for e in networkEdges:
            if e[0]==u:
                alt = dist[str(u)]+1
                if alt < dist[str(e[1])]:
                    dist[str(e[1])]=alt
                    prev[str(e[1])]=u
    return None, None
