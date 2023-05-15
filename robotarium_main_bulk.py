import functools
import os
import sys
# from tkinter import *
# from PIL import ImageTk,Image
import numpy as np
import pandas as pd
import random
import time
import openpyxl
import threading
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from rps.utilities.graph import *

from robotarium_init import *

#9288012
#4527271
#3729391
#1272922
#5362782
#2893817
#7828172
#1093802
#8199829
#6545288
# random.seed(12345)

# Direction radians
NORTH = 1.5708
SOUTH = 4.71239
WEST = 3.1419
EAST = 0.0000

BASE_DIR = [0, 1]

wait_time = 0

totalCost = 0

# rows, cols, A, numTasks, k, psi, centralized, visualizer, wall_prob, \
# seed, collisions, exp_strat, only_base_policy, verbose, depots = getParameters()

A, numTasks, k, psi, wall_prob, seed, only_base_policy, exp_seed = getParameters()

centralized = False
visualizer = False
collisions = False
exp_strat = 0
rows = 9
cols = 9
size = 9
step = 0.2
verbose = '-1'

new_data = {'Centralized':str(centralized), 'Seed #': str(seed), 'Exp Seed #': str(exp_seed),
            'Rows': str(rows), 'Cols': str(cols), 'Wall Prob': str(wall_prob),
            '# of Agents': str(A), '# of Tasks': str(numTasks), 'k': str(k),
            'psi': str(psi), 'Only Base Policy': str(only_base_policy)}




"""
if visualizer:
    root = Tk()
    root.resizable(height=None, width=None)
    root.title("RL Demo")

gridLabels=[]
pauseFlag=False
memSize=10

agentImages=[]
edgeList=[]
"""
vertices=[]
global colors
colors=[]
for i in range(100):
    colors+=[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,
                19,20,21,22,23,24,25,26,27,28,29,30,31]
colorIndex = ['#FDD835',
 '#E53935',
 '#03A9F4',
 '#78909C',
 '#8BC34A',
 '#7E57C2',
 '#26C6DA',
 '#827717',
 '#9C27B0',
 '#8D6E63',
 'none',
 '#FDD835',
 '#E53935',
 '#03A9F4',
 '#78909C',
 '#8BC34A',
 '#7E57C2',
 '#26C6DA',
 '#827717',
 '#9C27B0',
 '#8D6E63',
 '#FDD835',
 '#E53935',
 '#03A9F4',
 '#78909C',
 '#8BC34A',
 '#7E57C2',
 '#26C6DA',
 '#827717',
 '#9C27B0',
 '#8D6E63',
 '#FDD835']

marker_shapes = ['none','s','o','P', 'H', 'X', 'D','*', '^','p', '1','s','o','*', '^','p','P', 'H', 'X', 'D', '1','s','o','*', '^','p','P', 'H', 'X', 'D', '1',]

"""
if visualizer:
    #agent images
    for i in range(31):
        img= Image.open('images/agent'+str(i+1)+'.png')
        img = img.resize((50, 50), Image.ANTIALIAS)
        img=ImageTk.PhotoImage(img)
        agentImages.append(img)

    blankLabel=Label(root, text="     ", borderwidth=6, 
                    padx=padding,pady=padding,relief="solid")

    # #Add Tasks
    taskList=[]

    for i in range(rows):
        row=[]
        row1=[]
        for j in range(cols):
            row1.append(0)
        gridLabels.append(row1)
"""

## loop till you get a valid grid...
print("Initializing... ")
# out, offlineTrainRes = ut.load_instance(rows, seed)
# if out == None:
#     sys.exit(1)

## Uncomment and use it directly from init
out = init_valid_grid(A, numTasks, wall_prob=wall_prob,
                      seed=seed, colis=collisions)

gridGraph = out['gridGraph']
adjList = out['adjList']
vertices = out['verts']
agentVertices = out['agnt_verts']
taskVertices = out['task_verts']
obstacles = out["obs_verts"]
edgeList = out["adjList"]

obs_dir = out["obs_dir"]

random.seed(exp_seed)

## Uncomment below for better graphics in robotarium
# x_obs = [-1.0, 1]
# for x in x_obs:
#     for y in np.arange(1.0, -1.2, -0.2):
#         obstacles.append((x,round(y,1)))
#         obs_dir.append(1)

# y_obs = [-1.0, 1]
# for y in y_obs:
#     for x in np.arange(0.8, -1.0, -0.2):
#         obstacles.append((round(x,1),y))

for vertex in vertices:
    assert (vertex,vertex) in adjList
## truncate task list to accomodate lesser number of tasks
assert len(taskVertices) >= numTasks
if len(taskVertices) != numTasks:
    delete_inds = random.sample(range(len(taskVertices)), 
                                len(taskVertices)-numTasks)
    tasks = [taskVertices[i] for i in range(len(taskVertices)) \
                if i not in delete_inds]
    taskVertices = tasks
assert len(taskVertices) == numTasks

print(gridGraph)
for i in range(len(taskVertices)):
    colors+=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                19,20,21,22,23,24,25,26,27,28,29,30,31]

N = len(vertices)
N = 2**(psi+1)

"""
if visualizer:
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i][j]==0:
                gridLabels[i][j]=Label(root, text="     ", borderwidth=6, 
                    bg='#333366', padx=padding,pady=padding,relief="solid")
                gridLabels[i][j].grid(row=i,column=j)

            else:
                gridLabels[i][j]=Label(root, text="     ", borderwidth=6, 
                    padx=padding,pady=padding,relief="solid")
                gridLabels[i][j].grid(row=i,column=j)
                # vertices.append((i,j))

    for i in range(numTasks):
        img= Image.open('images/task.png')
        img = img.resize((50, 50), Image.ANTIALIAS)
        img=ImageTk.PhotoImage(img)
        taskList.append(img)

    for i in range(numTasks):
        gridGraph[taskVertices[i][0]][taskVertices[i][1]]='T'
        gridLabels[taskVertices[i][0]][taskVertices[i][1]].grid_forget()
        gridLabels[taskVertices[i][0]][taskVertices[i][1]]=Label(
            image=taskList[i],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[taskVertices[i][0]][taskVertices[i][1]].grid(
            row=taskVertices[i][0],column=taskVertices[i][1])
"""

"""
def changeCell(x,y,cellType,agentNum):
    sys.stdout.flush()
    gridLabels[x][y].grid_forget()
    sys.stdout.flush()

    if cellType=='task':
        gridLabels[x][y]=Label(image=taskList[0],borderwidth=6, 
            padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=r1,column=r2)
    elif cellType=='agent':
        gridLabels[x][y]=Label(image=agentImages[agentNum-1],borderwidth=6, 
            padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)
    elif cellType=='blank':
        gridLabels[x][y]=Label(root, text="     ", borderwidth=6, 
            padx=padding,pady=padding,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)
"""

def getTasksWithinRadius(agent_pos, taskVertices, radius):
    visible_tasks = []
    for t in taskVertices:
        if np.linalg.norm(np.asarray(t)-agent_pos)<=radius:
            visible_tasks.append(t)
    return visible_tasks

def bfs(vertices, edges, root, goal):
    Q = []
    labels = {}
    for v in vertices:
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
    Q.append(root)
    labels[str(root)] = True
    while (len(Q)) > 0:
        v = Q.pop(0)
        if v == goal:
            return True
        for e in edges:
            if e[0] == v:
                if e[1][0] == 0 and e[1][1] == 0:
                    tup = (0.0, 0.0)
                elif e[1][0] == 0:
                    tup = (0.0, e[1][1])
                elif e[1][1] == 0:
                    tup = (e[1][0], 0.0)
                else:
                    tup = (e[1][0], e[1][1])
                if labels[str(tup)] == False:
                    labels[str(tup)] = True
                    Q.append(tup)
    return False

def bfsNearestTask(networkVertices, networkEdges, source, taskVertices):
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
                    if tup in taskVertices:
                        t = tup
                        if prev[str(t)] != None or t==source:
                            path = []
                            while t != None:
                                path.append(t)
                                t = prev[str(t)]
                                dist += 1
                        return dist, list(reversed(path))

    return None,None

def dirShortestPath(networkVertices,networkEdges,source,target):
    Q=[]
    dist={}
    prev={}

    assert target in networkVertices
    assert source in networkVertices

    for v in networkVertices:
        if v[0] == 0 and v[1] == 0:
            tup = (0.0, 0.0)
        elif v[0] == 0:
            tup = (0.0, v[1])
        elif v[1] == 0:
            tup = (v[0], 0.0)
        else:
            tup = v
        dist[str(tup)]= 9999999999
        prev[str(tup)]=None
        Q.append(tup)
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
            if target[0] == 0 and target[1] == 0:
                t0 = (0.0, 0.0)
            elif target[0] == 0:
                t0 = (0.0, target[1])
            elif target[1] == 0:
                t0 = (target[0], 0.0)
            else:
                t0 = target
            t = t0
            if prev[str(t)] != None or t==source:
                while t != None:
                    S.append(t)
                    t=prev[str(t)]
                return dist[str(t0)],list(reversed(S))
        for e in networkEdges:
            if e[0]==u:
                if e[1][0] == 0 and e[1][1] == 0:
                    tup = (0.0, 0.0)
                elif e[1][0] == 0:
                    tup = (0.0, e[1][1])
                elif e[1][1] == 0:
                    tup = (e[1][0], 0.0)
                else:
                    tup = (e[1][0], e[1][1])
                alt = dist[str(u)]+1
                if alt < dist[str(tup)]:
                    dist[str(tup)]=alt
                    prev[str(tup)]=u
    return None, None
class Agent:
    def __init__(self,x,y,orient,ID,color=1):
        self.posX=x
        self.posY=y
        self.orientation = orient
        self.prev_move = None
        self.cost=0
        # if visualizer:
        #     changeCell(x,y,'agent',color)
        self.color=color
        self.ID=ID

        self.copy_number = 1

        self.posX_prime = x
        self.posY_prime = y
        self.cost_prime = 0
        self.color_prime = color

        # self.gui_split = False

        self.exploring = False
        self.exp_dir = ''
        self.exp_dist_remaining = 0

        self.reset()

    def resetColor(self):
        self.color=11
        self.color_prime=11
        # if visualizer:
        #     changeCell(self.posX,self.posY,'agent',self.color)
        sys.stdout.flush()

    def deallocate(self):
        del self.clusterID, self.children, self.stateMem, self.viewEdges
        del self.viewVertices, self.viewAgents, self.viewTasks, self.clusterVertices
        del self.clusterEdges, self.clusterTasks, self.clusterAgents
        del self.localVertices, self.localTasks, self.localEdges, self.localAgents
        del self.childMarkers, self.moveList

        del self.clusterID_prime, self.children_prime, self.stateMem_prime, self.viewEdges_prime
        del self.viewVertices_prime, self.viewAgents_prime, self.viewTasks_prime, self.clusterVertices_prime
        del self.clusterEdges_prime, self.clusterTasks_prime, self.clusterAgents_prime
        del self.localVertices_prime, self.localTasks_prime, self.localEdges_prime, self.localAgents_prime
        del self.childMarkers_prime, self.moveList_prime

    def reset(self):

        self.dir=''
        self.clusterID=[]
        self.parent=None
        self.children=[]
        self.stateMem=[]
        self.viewEdges=set()
        self.viewVertices=set([(self.posX,self.posY)])
        self.viewAgents=set()
        self.viewTasks=set()
        self.eta=0
        self.message=False
        self.clusterVertices=set()
        self.clusterEdges=set()
        self.clusterTasks=set()
        self.clusterAgents={}
        self.localVertices=set()
        self.localEdges=set()
        self.localTasks=set()
        self.localAgents=set()
        self.childMarkers=[]
        self.xOffset=0
        self.yOffset=0
        self.resetCounter=0
        self.moveList=dict()
        self.marker=False
        self.dfsNext=False

        self.dir_prime=''
        self.clusterID_prime=[]
        self.parent_prime=None
        self.children_prime=[]
        self.stateMem_prime=[]
        self.viewEdges_prime=set()
        self.viewVertices_prime=set([(self.posX,self.posY)])
        self.viewAgents_prime=set()
        self.viewTasks_prime=set()
        self.eta_prime=0
        self.message_prime=False
        self.clusterVertices_prime=set()
        self.clusterEdges_prime=set()
        self.clusterTasks_prime=set()
        self.clusterAgents_prime={}
        self.localVertices_prime=set()
        self.localEdges_prime=set()
        self.localTasks_prime=set()
        self.localAgents_prime=set()
        self.childMarkers_prime=[]
        self.xOffset_prime=0
        self.yOffset_prime=0
        self.resetCounter_prime=0
        self.moveList_prime=dict()
        self.marker_prime=False
        self.dfsNext_prime=False

    def setXPos(self,x):
        self.posX=x

    def setYPos(self,y):
        self.posY=y
    
    def setOrientation(self, orient):
        self.orientation = orient

    def getCostIncurred(self):
        return self.cost

    def getDir(self):
        return self.dir

    def move(self,dir):
        self.dir=dir

    def getColor(self):
        return self.color
    
    def getCluster(self):
        return self.clusterID

    def updateView(self, poses):
        self.viewEdges = set()
        self.viewEdges.add(((self.posX, self.posY), (self.posX, self.posY)))
        self.viewVertices = set([(self.posX, self.posY)])
        self.viewAgents = set()
        self.viewTasks = set()

        self.viewEdges_prime = set()
        self.viewEdges_prime.add(
            ((self.posX, self.posY), (self.posX, self.posY)))
        self.viewVertices_prime = set([(self.posX, self.posY)])
        self.viewAgents_prime = set()
        self.viewTasks_prime = set()
        # Create Internal Representation
        # k = hops
        # for each hop
        agent_pos = poses[:2,self.ID-1]

        visible_tasks = getTasksWithinRadius(agent_pos, taskVertices, step*k+(step*0.25))

        for i in range(1, k+1):
            # One hop in each direction
            up = round(self.posY+i*step, 1)
            down = round(self.posY-i*step, 1)
            left = round(self.posX-i*step, 1)
            right = round(self.posX+i*step, 1)
            # check if it exist in the grid bounds
            if (right, self.posY) in vertices:
                # self.viewEdges.add(((self.posX,self.posY),(self.posX+i,self.posY)))
                # self.viewEdges.add(((self.posX+i,self.posY),(self.posX,self.posY)))
                self.viewVertices.add((right, self.posY))
                self.viewVertices_prime.add((right, self.posY))
                if any([np.linalg.norm(np.asarray(t)-np.asarray((right, self.posY)))<=0.1 for t in visible_tasks]):
                    self.viewTasks.add((right, self.posY))
                    self.viewTasks_prime.add((right, self.posY))

                # for remaining hops
                for j in range(1, k-i+1):
                    up_steps = round(self.posY+j*step, 1)
                    down_steps = round(self.posY-j*step, 1)
                    if (right, up_steps) in vertices:
                        # self.viewEdges.add(((right,up_steps-1),(right,up_steps)))
                        # self.viewEdges.add(((right,up_steps),(right,up_steps-1)))
                        self.viewVertices.add((right, up_steps))
                        self.viewVertices_prime.add((right, up_steps))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((right, up_steps)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((right, up_steps))
                            self.viewTasks_prime.add(
                                (right, up_steps))
                    if (right, down_steps) in vertices:
                        # self.viewEdges.add(((right,down_steps+1),(right,down_steps)))
                        # self.viewEdges.add(((right,down_steps),(right,down_steps+1)))
                        self.viewVertices.add((right, down_steps))
                        self.viewVertices_prime.add((right, down_steps))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((right, down_steps)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((right, down_steps))
                            self.viewTasks_prime.add(
                                (right, down_steps))

            # check if it exist in the grid bounds
            if (left, self.posY) in vertices:
                # self.viewEdges.add(((self.posX,self.posY),(left,self.posY)))
                # self.viewEdges.add(((left,self.posY),(self.posX,self.posY)))
                self.viewVertices.add((left, self.posY))
                self.viewVertices_prime.add((left, self.posY))
                if any([np.linalg.norm(np.asarray(t)-np.asarray((left, self.posY)))<=0.1 for t in visible_tasks]):
                    self.viewTasks.add((left, self.posY))
                    self.viewTasks_prime.add((left, self.posY))

                for j in range(1, k-i+1):
                    up_steps = round(self.posY+j*step, 1)
                    down_steps = round(self.posY-j*step, 1)
                    if (left, up_steps) in vertices:
                        # self.viewEdges.add(((left,up_steps-1),(left,up_steps)))
                        # self.viewEdges.add(((left,up_steps),(left,up_steps-1)))
                        self.viewVertices.add((left, up_steps))
                        self.viewVertices_prime.add((left, up_steps))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((left, up_steps)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((left, up_steps))
                            self.viewTasks_prime.add(
                                (left, up_steps))

                    if (left, down_steps) in vertices:
                        # self.viewEdges.add(((left,down_steps+1),(left,down_steps)))
                        # self.viewEdges.add(((left,down_steps),(left,down_steps+1)))
                        self.viewVertices.add((left, down_steps))
                        self.viewVertices_prime.add((left, down_steps))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((left, down_steps)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((left, down_steps))
                            self.viewTasks_prime.add(
                                (left, down_steps))

            # check if it exist in the grid bounds
            if (self.posX, up) in vertices:
                # self.viewEdges.add(((self.posX,self.posY),(self.posX,up)))
                # self.viewEdges.add(((self.posX,up),(self.posX,self.posY)))
                self.viewVertices.add((self.posX, up))
                self.viewVertices_prime.add((self.posX, up))
                if any([np.linalg.norm(np.asarray(t)-np.asarray((self.posX, up)))<=0.1 for t in visible_tasks]):
                    self.viewTasks.add((self.posX, up))
                    self.viewTasks_prime.add((self.posX, up))
                for j in range(1, k-i+1):
                    right_steps = round(self.posX+j*step, 1)
                    left_steps = round(self.posX-j*step, 1)
                    if (right_steps, up) in vertices:
                        # self.viewEdges.add(((right_steps-1,up),(right_steps,up)))
                        # self.viewEdges.add(((right_steps,up),(right_steps-1,up)))
                        self.viewVertices.add((right_steps, up))
                        self.viewVertices_prime.add((right_steps, up))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((right_steps, up)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((right_steps, up))
                            self.viewTasks_prime.add(
                                (right_steps, up))
                    if (left_steps, up) in vertices:
                        # self.viewEdges.add(((left_steps+1,up),(left_steps,up)))
                        # self.viewEdges.add(((left_steps,self.posY+1),(left_steps+1,up)))
                        self.viewVertices.add((left_steps, up))
                        self.viewVertices_prime.add((left_steps, up))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((left_steps, up)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((left_steps, up))
                            self.viewTasks_prime.add(
                                (left_steps, up))

            # check if it exist in the grid bounds
            if (self.posX, down) in vertices:
                # self.viewEdges.add(((self.posX,self.posY),(self.posX,self.posY-i)))
                # self.viewEdges.add(((self.posX,self.posY-i),(self.posX,self.posY)))
                self.viewVertices.add((self.posX, down))
                self.viewVertices_prime.add((self.posX, down))
                if any([np.linalg.norm(np.asarray(t)-np.asarray((self.posX, down)))<=0.1 for t in visible_tasks]):
                    self.viewTasks.add((self.posX, down))
                    self.viewTasks_prime.add((self.posX, down))
                for j in range(1, k-i+1):
                    right_steps = round(self.posX+j*step, 1)
                    left_steps = round(self.posX-j*step, 1)
                    if (right_steps, down) in vertices:
                        # self.viewEdges.add(((right_steps-1,down),(right_steps,down)))
                        # self.viewEdges.add(((right_steps,down),(right_steps-1,down)))
                        self.viewVertices.add((right_steps, down))
                        self.viewVertices_prime.add((right_steps, down))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((right_steps, down)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((right_steps, down))
                            self.viewTasks_prime.add(
                                (right_steps, down))
                    if (left_steps, down) in vertices:
                        # self.viewEdges.add(((left_steps+1,down),(left_steps,down)))
                        # self.viewEdges.add(((left_steps,self.posY+1),(left_steps+1,down)))
                        self.viewVertices.add((left_steps, down))
                        self.viewVertices_prime.add((left_steps, down))
                        if any([np.linalg.norm(np.asarray(t)-np.asarray((left_steps, down)))<=0.1 for t in visible_tasks]):
                            self.viewTasks.add((left_steps, down))
                            self.viewTasks_prime.add(
                                (left_steps, down))

            for u in self.viewVertices:
                up_edge = round(u[1]+1*step, 1)
                down_edge = round(u[1]-1*step, 1)
                left_edge = round(u[0]-1*step, 1)
                right_edge = round(u[0]+1*step, 1)
                if (right_edge, u[1]) in self.viewVertices:
                    self.viewEdges.add(((u[0], u[1]), (right_edge, u[1])))
                    self.viewEdges.add(((right_edge, u[1]), (u[0], u[1])))

                if (left_edge, u[1]) in self.viewVertices:
                    self.viewEdges.add(((u[0], u[1]), (left_edge, u[1])))
                    self.viewEdges.add(((left_edge, u[1]), (u[0], u[1])))

                if (u[0], up_edge) in self.viewVertices:
                    self.viewEdges.add(((u[0], u[1]), (u[0], up_edge)))
                    self.viewEdges.add(((u[0], up_edge), (u[0], u[1])))

                if (u[0], down_edge) in self.viewVertices:
                    self.viewEdges.add(((u[0], u[1]), (u[0], down_edge)))
                    self.viewEdges.add(((u[0], down_edge), (u[0], u[1])))

            for u in self.viewVertices_prime:
                up_edge = round(u[1]+1*step, 1)
                down_edge = round(u[1]-1*step, 1)
                left_edge = round(u[0]-1*step, 1)
                right_edge = round(u[0]+1*step, 1)
                if (right_edge, u[1]) in self.viewVertices_prime:
                    self.viewEdges_prime.add(
                        ((u[0], u[1]), (right_edge, u[1])))
                    self.viewEdges_prime.add(
                        ((right_edge, u[1]), (u[0], u[1])))

                if (left_edge, u[1]) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0], u[1]), (left_edge, u[1])))
                    self.viewEdges_prime.add(((left_edge, u[1]), (u[0], u[1])))

                if (u[0], up_edge) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0], u[1]), (u[0], up_edge)))
                    self.viewEdges_prime.add(((u[0], up_edge), (u[0], u[1])))

                if (u[0], down_edge) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0], u[1]), (u[0], down_edge)))
                    self.viewEdges_prime.add(((u[0], down_edge), (u[0], u[1])))

            s = self.viewVertices.copy()
            E = self.viewEdges.copy()
            T = self.viewTasks.copy()

            s_prime = self.viewVertices_prime.copy()
            E_prime = self.viewEdges_prime.copy()
            T_prime = self.viewTasks_prime.copy()
            # print(E)
            for u in s:
                if u != (self.posX, self.posY) and not bfs(s, E, (self.posX, self.posY), u):
                    for e in E:
                        if (e[0] == u or e[1] == u) and e in self.viewEdges:
                            self.viewEdges.remove(e)
                    self.viewVertices.remove(u)
                    if u in T:
                        self.viewTasks.remove(u)
            del s

            for u in s_prime:
                if u != (self.posX, self.posY) and not bfs(s_prime, E_prime, (self.posX, self.posY), u):
                    for e in E_prime:
                        if (e[0] == u or e[1] == u) and e in self.viewEdges_prime:
                            self.viewEdges_prime.remove(e)
                    self.viewVertices_prime.remove(u)
                    if u in T_prime:
                        self.viewTasks_prime.remove(u)
            del s_prime

    def mapOffset(self,offX,offY,mapVerts,mapEdges,mapTasks,mapAgents):
        vertices = set()
        edges = set()
        taskSet = set()
        agentSet = {}
        for v in mapVerts:
            vertices.add((round(v[0]-offX, 1), round(v[1]-offY, 1)))
        for e in mapEdges:
            newEdge = ((round(e[0][0]-offX, 1), round(e[0][1]-offY, 1)),
                       (round(e[1][0]-offX, 1), round(e[1][1]-offY, 1)))
            edges.add(newEdge)
        for t in mapTasks:
            taskSet.add((round(t[0]-offX, 1), round(t[1]-offY, 1)))
        for a in mapAgents:
            agentSet[a.ID] = (offX, offY)
            # agent.posX-=offX
            # agent.posY-=offY
            # agentSet.add(agent)

        #print("mapOffset: ", agentSet)
        return [vertices,edges,taskSet,agentSet]

    def updateLocalView(self):
        l=self.mapOffset(self.posX,self.posY,self.viewVertices,self.viewEdges,self.viewTasks,self.viewAgents)
        self.localVertices=l[0]
        self.localEdges=l[1]
        self.localTasks=l[2]
        self.localAgents=l[3]

def updateAgentToAgentView(poses):
    # changed to using vectors based on sensor readings
    ## why can't this be done for each agent ? becoz some other agents might move later into the view
    for a in agents:
        a.viewAgents.add(a)
        a.viewAgents_prime.add(a)
        # print("Agent ", a.ID-1," view:", )
        for idx in delta_disk_neighbors(poses, a.ID-1, step*k+(step*0.25)):
            # dxi = poses[:2,[a.ID-1]] - poses[:2,[idx]]
            # norm_ = np.linalg.norm(dxi)
            # angle = angle_between(np.array(NORTH), poses[:2,[idx]])
            for v in a.viewVertices:
                dxi = np.reshape(np.array(v), (2,1)) - poses[:2,[idx]]
                # print("distance",dxi)
                # print(poses[:2, [idx]])
                # print()
                if np.linalg.norm(dxi) < 0.1:
                    a.viewAgents.add(agents[idx])
                    break
            
            for v in a.viewVertices_prime:
                dxi = np.reshape(np.array(v), (2,1)) - poses[:2,[idx]]
                if np.linalg.norm(dxi) < 0.1:
                    a.viewAgents_prime.add(agents[idx])
                    break

id = 1
agents = []
for i in range(A):
    agent = Agent(agentVertices[i][0], agentVertices[i][1], NORTH, id, 11)
    agents.append(agent)
    # print("{}: ({},{})".format(agent.ID, agent.posX, agent.posY))
    id += 1

pos = []
for a in agents:
    pos.append([a.posX, a.posY, a.orientation])

starting_pos = np.asarray(pos).T

# print(starting_pos)
r_env = robotarium.Robotarium(number_of_robots=A, show_figure=False, initial_conditions=starting_pos,sim_in_real_time=True)

## Uncomment below when doing an actual experiment on robotarium with graphics

# marker_size_obs = determine_marker_size(r_env, 0.03)
# marker_obs_sml = determine_marker_size(r_env, 0.02)
# marker_size_goal = determine_marker_size(r_env, 0.02)
# marker_size_robot = determine_marker_size(r_env, 0.04)
# taskss = [r_env.axes.scatter(taskVertices[ii][0], taskVertices[ii][1], s=marker_size_goal, marker='o', facecolors='y',edgecolors='none',linewidth=2,zorder=-2)
# for ii in range(len(taskVertices))]

# horizontal_obs = [[-3, -1], [3, -1], [3, 1], [-3, 1], [-3, -1]]
# vert_obs = [[-1, -3], [1, -3], [1, 3], [-1, 3], [-1, -3]]

# square = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]

# print(len(obs_dir))
# print(len(obstacles))

# shapes_dir = {
#     0: horizontal_obs,
#     1: vert_obs,
#     2: 's'
# }
# 0 - horizontal
# 1 - vertical
# 2 - square

# obs = [r_env.axes.scatter(obstacles[ii][0], obstacles[ii][1], s= marker_obs_sml if obs_dir[ii] == 2 else marker_size_obs, marker=shapes_dir[obs_dir[ii]], facecolors='k',edgecolors='k',linewidth=5,zorder=-2)
# for ii in range(len(obstacles))]

## New obs
# obs = [r_env.axes.scatter(obstacles[ii][0], obstacles[ii][1], s= marker_size_obs, marker=shapes_dir[2], facecolors='k',edgecolors='k',linewidth=5,zorder=-2)
# for ii in range(len(obstacles))]


robot_markers = []

# unicycle_pose_controller = my_controller2(1, 0.5, 0.2, np.pi, 0.05, 0.03, 0.01)
# unicycle_pose_controller = create_clf_unicycle_pose_controller()
##og
# unicycle_pose_controller = create_hybrid_unicycle_pose_controller(1, 0.5, 0.2, np.pi, 0.05, 0.03, 0.01)
## take it slow there are still some collisions occuring for linear_velocity_gain = 0.3
unicycle_pose_controller = create_hybrid_unicycle_pose_controller(0.3, 0.3, 0.2, np.pi, 0.05, 0.03, 0.01)
# unicycle_pose_controller = create_hybrid_unicycle_pose_controller()
# unicycle_pose_controller = create_clf_unicycle_position_controller(linear_velocity_gain=0.5, angular_velocity_gain=1)

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate(100, 0.12, 0.01, 0.2)
# uni_barrier_cert = create_unicycle_barrier_certificate(100, 0.15, 0.05, 0.2)
# uni_barrier_cert = create_unicycle_barrier_certificate()

def getOppositeDirection(direction):
    if direction == 'n':
        return 's'
    elif direction == 's':
        return 'n'
    elif direction == 'e':
        return 'w'
    elif direction == 'w':
        return 'e'
    elif direction == 'q':
        return 'q'

def getExplorationMove(agent, updated_pos):
    legal = getLegalMovesFrom(agent, updated_pos)
    if exp_strat == 0:
        if len(legal) > 1:
            legal.remove('q')
        return random.choice(legal)

    """
    elif exp_strat == -1:
        pos = (agent.posX, agent.posY)
        shortestDist = float('inf')
        exp_move = None
        for v in taskVertices:
            try:
                dist = lookupTable[str(pos)][str(v)][0]
                path = lookupTable[str(pos)][str(v)][1]
                if dist < shortestDist:
                    shortestDist = dist
                    exp_move = path[1]
            except KeyError:
                continue

        # assert exp_move != None
        if exp_move == (agent.posX+1, agent.posY):
            return 'e'
        elif exp_move == (agent.posX-1, agent.posY):
            return 'w'
        elif exp_move == (agent.posX, agent.posY-1):
            return 's'
        elif exp_move == (agent.posX, agent.posY+1):
            return 'n'
        elif exp_move == None:
            return 'q'

    elif exp_strat >= 1:
        ## randomly sample a direction and a distance in that direction
        ## max distance is given by value of exp_strat
        if len(legal) > 1:
            legal.remove('q')
        print(legal, agent.exploring, agent.exp_dir)
        if not agent.exploring:
            agent.exploring = True

            distance = random.randint(1, exp_strat)
            direction = random.choice(legal)

            agent.exp_dist_remaining = distance
            agent.exp_dir = direction

        else:
            if agent.exp_dir in legal:
                direction = agent.exp_dir
            else:
                opp_old_dir = getOppositeDirection(agent.exp_dir)
                if (len(legal) > 1):
                    legal.remove(opp_old_dir)
                print(legal)
                direction = random.choice(legal)

                agent.exp_dir = direction ## Maybe we don't want to go back to where we came from?

        print("EXP_FLAG: ", seed, agent.exploring, agent.exp_dir, agent.exp_dist_remaining)

        agent.exp_dist_remaining -= 1
        if agent.exp_dist_remaining == 0:
            agent.exploring = False

        return direction
    """

## Guaranteed to move
def getLegalMovesFrom(agent, updated_pos):
    # Guard against potential collision
    moves = ['q']
    ## NOTNEEDED: all the agents in the view and ones which have moved before
    ## use updated_pos instead of a.viewAgents since that is outdated
    # filtered = [x for x in a.viewAgents if x.ID < agent.ID]
    # filtered_pos = []
    # for n in filtered:
    #     filtered_pos.append((n.posX, n.posY))
    ## TODO: Rewrite this copy pasted crap, this is not right, why are there 2 hops ?
    up = round(agent.posY+1*step, 1)
    down = round(agent.posY-1*step, 1)
    left = round(agent.posX-1*step, 1)
    right = round(agent.posX+1*step, 1)

    # check if it exist in the grid bounds
    if (right, agent.posY) in vertices and not updated_pos.get((right, agent.posY)):
        moves.append('e')
    # check if it exist in the grid bounds
    if (left, agent.posY) in vertices and not updated_pos.get((left, agent.posY)):
        moves.append('w')
    # check if it exist in the grid bounds
    if (agent.posX, up) in vertices and not updated_pos.get((agent.posX, up)):
        moves.append('n')
    # check if it exist in the grid bounds
    if (agent.posX, down) in vertices and not updated_pos.get((agent.posX, down)):
        moves.append('s')
    return moves

def executeStep(r_pos, eAgnt, goal_points):
    if (np.size(at_pose(r_pos[:, eAgnt], goal_points[:,eAgnt])) != A):
        # Get poses of agents
        r_pos = r_env.get_poses()
        ## Remove Tasks
        remove_t = set()
        for t in taskVertices:
            if any(np.linalg.norm(r_pos[:2,:] - np.reshape(np.array(t), (2,1)), axis=0) < 0.1):
                remove_t.add(t)
                # for ts in taskss:
                #     if ts.get_offsets()[0][0] == t[0] and ts.get_offsets()[0][1] == t[1]:
                #         ts.set_visible(False)
                #         break
        for t in remove_t:
            taskVertices.remove(t)
        # Create unicycle control inputs
        dxu = unicycle_pose_controller(r_pos, goal_points)
        # Create safe control inputs (i.e., no collisions)
        dxu = uni_barrier_cert(dxu, r_pos)
        # Set the velocities
        r_env.set_velocities(np.arange(A), dxu)
        # Iterate the simulation
        r_env.step()
        return r_pos
    else:
        r_pos = r_env.get_poses()
        r_env.step()
        return r_pos

def movePrecedenceCmp(agentA, agentB):
    ## simple precedence filter based on remaining actual moves in the move list
    ## Filters out even the wait moves in between, but ideally should remove only the trailing wait moves
    agentA_filtMovesLen = len([x for x in agentA.moves if x != 'q'])
    agentB_filtMovesLen = len([x for x in agentB.moves if x != 'q'])
    return agentB_filtMovesLen - agentA_filtMovesLen
    # return -1 if swap A before B
    # return 0 if no swap
    # return 1 if swap B before A

def stateUpdate(r_pos, totalCost, waitCost, explore_steps, tempTasks):

    sys.stdout.flush()
    ## First resolve collision between agents in clusters and have them move first

    all_pos = {}
    prev_pos = {}
    agentsInClusterLen = 0

    for a in agents:
        a.back = False
        if len(a.clusterID)!=0:
            ## Add the pos to old pos_directory
            if prev_pos.get((a.posX, a.posY)):
                prev_pos[(a.posX, a.posY)].append(a.ID)
            else:
                prev_pos[(a.posX, a.posY)] = [a.ID]

            agentsInClusterLen += 1
            ## get the next move for an agent and account for costs
            try:
                a.dir = a.moves.pop(0)
                if a.dir != 'q':
                    totalCost += 1
                else:
                    waitCost += 1
            except IndexError:
                a.dir = 'q'
                waitCost += 1
    
            if a.getDir() == 'e':
                east_move = round(a.posX+1*step, 1)
                a.setXPos(east_move)
                a.setOrientation(EAST)
            elif a.getDir() == 'w':
                west_move = round(a.posX-1*step, 1)
                a.setXPos(west_move)
                a.setOrientation(WEST)
            elif a.getDir() == 's':
                south_move = round(a.posY-1*step, 1)
                a.setYPos(south_move)
                a.setOrientation(SOUTH)
            elif a.getDir() == 'n':
                north_move = round(a.posY+1*step, 1)
                a.setYPos(north_move)
                a.setOrientation(NORTH)
            elif a.getDir() == 'q':
                pass
            else:
                raise ValueError("Incorrect direction. ")
            
            ## Add the new pos to updated pos_directory
            if all_pos.get((a.posX, a.posY)):
                all_pos[(a.posX, a.posY)].append(a.ID)
            else:
                all_pos[(a.posX, a.posY)] = [a.ID]
    
    ## TODO: Resolve collision - both 1. Intra Cluster 2. Inter Cluster
    ## Resolve all collisions
    ## NOTE: Backtracking can cause collisions as well and for trajectory based forward moving this is even bigger
    ## problem because then we would have to backtrack for mutliple steps
    ## SOLN: Don't move down trajectory unless completely sure
    ## TODO: Check for deadlocks that can occur: x -> y; y -> x || x    j   y || j is waiting permanently
    ## this occurs since trajectories don't allow to move into previously occupied location
    ## Account for exploring agents which are waiting
    for a in agents:
        if len(a.clusterID)==0:
            a.updateView(r_pos)
            found = False
            for vertex in a.viewVertices:
                if vertex in tempTasks:
                    found = True
                    break
            if found == True:
                a.dir = 'q'
                waitCost += 1

    for a in agents:
        if len(a.clusterID)==0 and a.dir == 'q':
            agentsInClusterLen += 1
            if all_pos.get((a.posX, a.posY)):
                all_pos[(a.posX, a.posY)].append(a.ID)
            else:
                all_pos[(a.posX, a.posY)] = [a.ID]

    while len(all_pos) < agentsInClusterLen:
        for pos in list(all_pos.keys()):
            ## check for collisions
            if len(all_pos[pos]) > 1:
                print("Colision at Loc:",pos, " for agents", all_pos[pos])
                colliding_agents_IDs = all_pos[pos]
                ## Find agents which were previously waiting here or its their original position and have back tracked
                ## vs agents which have moved into this new position and have collided with already residing agents
                colliding_agents_moved = list(filter(lambda a: a.ID in colliding_agents_IDs and len(a.clusterID) > 0 and (a.dir != 'q' and not a.back), agents))
                prev_waiting = False
                if len(colliding_agents_moved) == len(colliding_agents_IDs):
                    # No previously waiting agents
                    prev_waiting = False
                elif len(colliding_agents_moved) == len(colliding_agents_IDs)-1:
                    # One previously waiting agents
                    prev_waiting = True
                else:
                    raise ValueError("Impossible, two agents cannot be waiting at same location")
                sorted_agents = sorted(colliding_agents_moved, key=functools.cmp_to_key(movePrecedenceCmp))
                curr_agnt = None
                if not prev_waiting:
                    ## Since no one was waiting the agent to travel farthest will retain its move
                    curr_agnt = sorted_agents.pop(0)
                else:
                    ## Find the currently waiting agent or one who backtracked
                    curr_agnt = list(filter(lambda a: a.ID in colliding_agents_IDs and (a.dir == 'q' or a.back), agents))[0]
                assert curr_agnt != None
                curr_always_waiting = len(curr_agnt.clusterID) == 0 or len([x for x in curr_agnt.moves if x != 'q']) == 0
                ## Now the current colliding location is filled and who occupies it is fixed
                ## For everyone else find a safe next location or backtrack
                for collide_agt in sorted_agents:
                    if len([x for x in collide_agt.moves if x != 'q']) > 0:
                        ## Trajectory agent
                        safe = False
                        any_future_safe = False ## used this later to make better waits for travelling agents
                        trajectory_seen = 0
                        agent_t_cost = 0
                        agent_w_cost = 0
                        agent_pos = pos
                        agent_orient = collide_agt.orientation
                        ## keep looking for next safe move until all of trajectory is seen
                        ## A safe move is one where no one currently is and no one can backtrack as well
                        while not safe and trajectory_seen < len(collide_agt.moves):
                            next_move = collide_agt.moves[trajectory_seen]
                            trajectory_seen += 1
                            if next_move != 'q':
                                agent_t_cost += 1
                            else:
                                agent_w_cost += 1
                            
                            if next_move == 'e':
                                east_move = round(agent_pos[0]+1*step, 1)
                                agent_pos = (east_move, agent_pos[1])
                                agent_orient = EAST
                            elif next_move == 'w':
                                west_move = round(agent_pos[0]-1*step, 1)
                                agent_pos = (west_move, agent_pos[1])
                                agent_orient = WEST
                            elif next_move == 's':
                                south_move = round(agent_pos[1]-1*step, 1)
                                agent_pos = (agent_pos[0], south_move)
                                agent_orient = SOUTH
                            elif next_move == 'n':
                                north_move = round(agent_pos[1]+1*step, 1)
                                agent_pos = (agent_pos[0], north_move)
                                agent_orient = NORTH
                            elif next_move == 'q':
                                pass
                            else:
                                raise ValueError("Incorrect direction. ")
                            
                            ## Make sure that it is really unsafe by checking if its currently occupied by someone or
                            ## is someone else's previous location and not just your own!
                            if all_pos.get(agent_pos) or (prev_pos.get(agent_pos) and not (collide_agt.ID in prev_pos.get(agent_pos))):
                                ## Not a safe location
                                ## Check if any the agent in the location will stay there permanently by looking at the length of their remaining moves?
                                ## if so not so then just wait for future ie. check if all the agents in the location are bound to move
                                ## condition read "if no agents in next location such that no move left -> conversely all agents in next location have aleast 1 move left"
                                if all_pos.get(agent_pos) and len([x for x in all_pos.get(agent_pos) if len([y for y in agents[x-1].moves if y != 'q']) == 0]) == 0:
                                    ## If there is only one agent then check its remaining moves length
                                    ## If there are multiple agents then see 
                                    any_future_safe = True
                            else:
                                ## Found safe then update move, location, trajectory and account for taken move costs
                                all_pos[agent_pos] = [collide_agt.ID]
                                all_pos[pos].remove(collide_agt.ID)
                                totalCost += agent_t_cost
                                waitCost += agent_w_cost
                                collide_agt.setXPos(agent_pos[0])
                                collide_agt.setYPos(agent_pos[1])
                                collide_agt.dir = next_move
                                collide_agt.setOrientation(agent_orient)
                                for _ in range(trajectory_seen):
                                    collide_agt.moves.pop(0)
                                safe = True
                                print("Collision resolution for Agent:", collide_agt.ID, " move to (",collide_agt.posX,",",collide_agt.posY,")")
                                break

                        ## Still can't find a safe location but in future a location might open up then wait
                        ## for future to move
                        if not safe and any_future_safe:
                            ## Backtrack and discount for previously computed costs
                            collide_agt.moves.insert(0, collide_agt.dir)
                            collide_agt.back = True
                            all_pos[pos].remove(collide_agt.ID)
                            if collide_agt.dir != 'q':
                                totalCost -= 1
                            else:
                                waitCost -= 1
                            opposite_dir = getOppositeDirection(collide_agt.dir)
                            if opposite_dir == 'e':
                                east_move = round(collide_agt.posX+1*step, 1)
                                collide_agt.setXPos(east_move)
                                collide_agt.setOrientation(EAST)
                            elif opposite_dir == 'w':
                                west_move = round(collide_agt.posX-1*step, 1)
                                collide_agt.setXPos(west_move)
                                collide_agt.setOrientation(WEST)
                            elif opposite_dir == 's':
                                south_move = round(collide_agt.posY-1*step, 1)
                                collide_agt.setYPos(south_move)
                                collide_agt.setOrientation(SOUTH)
                            elif opposite_dir == 'n':
                                north_move = round(collide_agt.posY+1*step, 1)
                                collide_agt.setYPos(north_move)
                                collide_agt.setOrientation(NORTH)
                            elif opposite_dir == 'q':
                                pass
                            else:
                                raise ValueError("Incorrect direction. ")
                            
                            print("No Collision resolution for Agent:", collide_agt.ID, " backtrack to (",collide_agt.posX,",",collide_agt.posY,") for now")

                            if all_pos.get((collide_agt.posX, collide_agt.posY)):
                                all_pos[(collide_agt.posX, collide_agt.posY)].append(collide_agt.ID)
                            else:
                                all_pos[(collide_agt.posX, collide_agt.posY)] = [collide_agt.ID]
                        ## All the locations in the trajectory will always be occupied by as final positions
                        ## therefore backtrack and call it good
                        elif not safe and not any_future_safe:
                            ## Just Backtrack
                            collide_agt.back = True
                            all_pos[pos].remove(collide_agt.ID)
                            opposite_dir = getOppositeDirection(collide_agt.dir)
                            if opposite_dir == 'e':
                                east_move = round(collide_agt.posX+1*step, 1)
                                collide_agt.setXPos(east_move)
                                collide_agt.setOrientation(EAST)
                            elif opposite_dir == 'w':
                                west_move = round(collide_agt.posX-1*step, 1)
                                collide_agt.setXPos(west_move)
                                collide_agt.setOrientation(WEST)
                            elif opposite_dir == 's':
                                south_move = round(collide_agt.posY-1*step, 1)
                                collide_agt.setYPos(south_move)
                                collide_agt.setOrientation(SOUTH)
                            elif opposite_dir == 'n':
                                north_move = round(collide_agt.posY+1*step, 1)
                                collide_agt.setYPos(north_move)
                                collide_agt.setOrientation(NORTH)
                            elif opposite_dir == 'q':
                                pass
                            else:
                                raise ValueError("Incorrect direction. ")
                            
                            print("No Collision resolution for Agent:", collide_agt.ID, " backtrack to (",collide_agt.posX,",",collide_agt.posY,") and stay (final pos)")
                            ## Since final move only going to wait hence forth
                            collide_agt.dir = 'q'
                            collide_agt.moves = []
                            if all_pos.get((collide_agt.posX, collide_agt.posY)):
                                all_pos[(collide_agt.posX, collide_agt.posY)].append(collide_agt.ID)
                            else:
                                all_pos[(collide_agt.posX, collide_agt.posY)] = [collide_agt.ID]
                    else:
                        ## Final move agent simply backtracks
                        #  but if current occupying agent is going to wait there indefinitely then backtrack but
                        #  account for the final move and call it good.
                        if curr_always_waiting:
                            ## Just Backtrack
                            collide_agt.back = True
                            all_pos[pos].remove(collide_agt.ID)
                            opposite_dir = getOppositeDirection(collide_agt.dir)
                            if opposite_dir == 'e':
                                east_move = round(collide_agt.posX+1*step, 1)
                                collide_agt.setXPos(east_move)
                                collide_agt.setOrientation(EAST)
                            elif opposite_dir == 'w':
                                west_move = round(collide_agt.posX-1*step, 1)
                                collide_agt.setXPos(west_move)
                                collide_agt.setOrientation(WEST)
                            elif opposite_dir == 's':
                                south_move = round(collide_agt.posY-1*step, 1)
                                collide_agt.setYPos(south_move)
                                collide_agt.setOrientation(SOUTH)
                            elif opposite_dir == 'n':
                                north_move = round(collide_agt.posY+1*step, 1)
                                collide_agt.setYPos(north_move)
                                collide_agt.setOrientation(NORTH)
                            elif opposite_dir == 'q':
                                pass
                            else:
                                raise ValueError("Incorrect direction. ")
                            
                            print("No Collision resolution for Agent:", collide_agt.ID, " backtrack to (",collide_agt.posX,",",collide_agt.posY,") and stay (final pos)")
                            ## Since final move only going to wait hence forth
                            collide_agt.dir = 'q'

                            if all_pos.get((collide_agt.posX, collide_agt.posY)):
                                all_pos[(collide_agt.posX, collide_agt.posY)].append(collide_agt.ID)
                            else:
                                all_pos[(collide_agt.posX, collide_agt.posY)] = [collide_agt.ID]

                        else:
                            ## Backtrack and discount for previously computed costs
                            collide_agt.moves.insert(0, collide_agt.dir)
                            collide_agt.back = True
                            all_pos[pos].remove(collide_agt.ID)
                            if collide_agt.dir != 'q':
                                totalCost -= 1
                            else:
                                waitCost -= 1
                            opposite_dir = getOppositeDirection(collide_agt.dir)
                            if opposite_dir == 'e':
                                east_move = round(collide_agt.posX+1*step, 1)
                                collide_agt.setXPos(east_move)
                                collide_agt.setOrientation(EAST)
                            elif opposite_dir == 'w':
                                west_move = round(collide_agt.posX-1*step, 1)
                                collide_agt.setXPos(west_move)
                                collide_agt.setOrientation(WEST)
                            elif opposite_dir == 's':
                                south_move = round(collide_agt.posY-1*step, 1)
                                collide_agt.setYPos(south_move)
                                collide_agt.setOrientation(SOUTH)
                            elif opposite_dir == 'n':
                                north_move = round(collide_agt.posY+1*step, 1)
                                collide_agt.setYPos(north_move)
                                collide_agt.setOrientation(NORTH)
                            elif opposite_dir == 'q':
                                pass
                            else:
                                raise ValueError("Incorrect direction. ")
                            
                            print("No Collision resolution for Agent:", collide_agt.ID, " backtrack to (",collide_agt.posX,",",collide_agt.posY,") for now")
                            if all_pos.get((collide_agt.posX, collide_agt.posY)):
                                all_pos[(collide_agt.posX, collide_agt.posY)].append(collide_agt.ID)
                            else:
                                all_pos[(collide_agt.posX, collide_agt.posY)] = [collide_agt.ID]

    ## Reset all the backtrack flags
    for a in agents:
        if len(a.clusterID)!=0:
            a.back = False
    
    ## Append to the all_pos of cluster agents which moved and place previous locations of exploring agents
    ## except exploring agents which now wait since they have seen a task and whoes locations already exists in all_pos
    ## in there as well for added prevention of swap moves
    for a in agents:
        if len(a.clusterID)==0 and a.dir != 'q':
            if all_pos.get((a.posX, a.posY)):
                all_pos[(a.posX, a.posY)].append(a.ID)
            else:
                all_pos[(a.posX, a.posY)] = [a.ID]

    for a in agents:
        if len(a.clusterID)==0:
            ## Not needed since done above
            # a.updateView(r_pos)
            found = False
            for vertex in a.viewVertices:
                if vertex in tempTasks:
                    found = True
                    break
            if found == False:
                a.dir=getExplorationMove(a, all_pos)
                if a.dir != 'q':
                    if verbose == 'x':
                        print(a.ID, a.dir)
                    totalCost+=1
                    explore_steps += 1
                else:
                    waitCost += 1
            
            if a.getDir() == 'e':
                east_move = round(a.posX+1*step, 1)
                a.setXPos(east_move)
                a.setOrientation(EAST)
            elif a.getDir() == 'w':
                west_move = round(a.posX-1*step, 1)
                a.setXPos(west_move)
                a.setOrientation(WEST)
            elif a.getDir() == 's':
                south_move = round(a.posY-1*step, 1)
                a.setYPos(south_move)
                a.setOrientation(SOUTH)
            elif a.getDir() == 'n':
                north_move = round(a.posY+1*step, 1)
                a.setYPos(north_move)
                a.setOrientation(NORTH)
            elif a.getDir() == 'q':
                pass
            else:
                raise ValueError("Incorrect direction. ")
            
            ## Add the new pos to updated pos_directory
            if all_pos.get((a.posX, a.posY)):
                all_pos[(a.posX, a.posY)].append(a.ID)
            else:
                all_pos[(a.posX, a.posY)] = [a.ID]

    # for a in agents:
    #     # if visualizer:
    #     #     changeCell(a.posX, a.posY, 'blank', 0)

        # if visualizer:
        #     changeCell(a.posX, a.posY, 'agent', a.color)

        ## Task should be removed when the actual motion happens
        # if (a.posX, a.posY) in taskVertices:
        #     taskVertices.remove((a.posX, a.posY))

        sys.stdout.flush()
    pos = np.zeros((3, A))
    for idx, a in enumerate(agents):
        pos[:,idx] = np.asarray([a.posX, a.posY, a.orientation])
        # pos.append()
    return pos, totalCost, waitCost, explore_steps

"""
def multiAgentRolloutCent(networkVertices,networkEdges,agents,taskPos,agent,prevMoves,lookupTable):
    currentPos={}
    for a in agents:
        currentPos[a]=(a.posX,a.posY)
    currentTasks=taskPos.copy()

    prevCost=0
    for a in agents:
        if a in prevMoves:
            if prevMoves[a]=='n':
                currentPos[a]=(currentPos[a][0],currentPos[a][1]+1)
            if prevMoves[a]=='s':
                currentPos[a]=(currentPos[a][0],currentPos[a][1]-1)
            if prevMoves[a]=='e':
                currentPos[a]=(currentPos[a][0]+1,currentPos[a][1])
            if prevMoves[a]=='w':
                currentPos[a]=(currentPos[a][0]-1,currentPos[a][1])
            if prevMoves[a]=='q':
                currentPos[a]=(currentPos[a][0],currentPos[a][1])
            if currentPos[a] in currentTasks:
                currentTasks.remove(currentPos[a])

    if len(currentTasks)==0:
        return 'q', 0

    assert len(currentTasks)>0
    minCost=float('inf')
    bestMove=None
    Qfactors=[]

    for e in networkEdges:
        if e[0]==(agent.posX,agent.posY):
            # #print('flag')
            tempCurrentTasks=currentTasks.copy()
            ##print(tempCurrentTasks)
            tempPositions=currentPos.copy()
            # #print(tempPositions)
            if e[1] != e[0]:
                cost=prevCost+1
            else:
                cost=prevCost ## not a bug!!
            # cost = prevCost + 1
            tempPositions[agent]=e[1]
            #print("lookahead edge: ", e)
            if tempPositions[agent]in tempCurrentTasks:
                tempCurrentTasks.remove(tempPositions[agent])
                #bestMove=e[1]
                #break
            # #print(tempCurrentTasks)
            rounds = 0
            while len(tempCurrentTasks)>0:
                if rounds > len(networkVertices) + len(networkEdges):
                    cost += 10_000
                    break
                # #print('flag2')
                # print("POS: ", tempPositions, tempCurrentTasks, cost)
                for a in agents:
                    if ((a.ID < agent.ID) and prevMoves[a] != 'q') or (a.ID >= agent.ID):
                        shortestDist=float('inf')
                        bestNewPos=None
                        #assert len(tempCurrentTasks)>0
                        #print("======== Agent a: {} ========".format((a.posX, a.posY)))
                        for t in tempCurrentTasks:
                            # #print("MR: ",tempPositions[agent], tempPositions[a], t)
                            # dist,path=bfShortestPath(networkVertices,networkEdges,tempPositions[a],t)
                            try:
                                #print("PRE:", tempPositions, a.posX, a.posY)
                                dist,path = (lookupTable[str(tempPositions[a])][str(t)])
                            except (AssertionError, KeyError):
                                continue
                            #dist, path = lookupTable[str(tempPositions[a])][str(t)]

                            # #print("\tPath: {}; Dist: {}".format(path, dist))
                            if dist<shortestDist:
                                shortestDist=dist
                                bestNewPos=path
                        if bestNewPos != None:
                            if (e[1] == (agent.posX, agent.posY)) and (a.ID == agent.ID):
                                ## if this is wait move, main agent must become inactive for
                                ## rest of cost-to-go computation
                                pass
                            else:
                                tempPositions[a]=bestNewPos
                        if tempPositions[a] in tempCurrentTasks:
                            tempCurrentTasks.remove(tempPositions[a])
                        if e[1] != (agent.posX, agent.posY) or (a.posX!=agent.posX) or (a.ID != agent.ID):
                            cost += 1
                        rounds += 1
                        if len(tempCurrentTasks)==0:
                            break
            #print("resulting cost for action {}: {} ".format((e[0],e[1]), cost))
            #print(e[1], cost)
            if cost<minCost:
                minCost=cost
                bestMove=e[1]
                if bestMove == e[0]:
                    print("Waiting! ")
                    # elif cost==minCost:
            #    r=random.random()
            #   if r<0.5:
            #      bestMove=e[1]
            Qfactors.append((e[1],cost))
            del tempPositions
            del tempCurrentTasks

    assert bestMove!=None
    minQ = float('inf')
    for factor in Qfactors:
        if factor[1] < minQ:
            minQ = factor[1]

    print(agent.posX, agent.posY, Qfactors)
    ## collect all ties...
    wait_ind = None
    ties = []

    for factor in Qfactors:
        if factor[1] == minQ:
            if factor[0]==(agent.posX+1,agent.posY):
                if (agent.prev_move == None) or (agent.prev_move != None and getOppositeDirection('e') != agent.prev_move):
                    ties.append((factor,'e'))
            elif factor[0]==(agent.posX-1,agent.posY):
                if ((agent.prev_move == None) or (agent.prev_move != None and getOppositeDirection('w') != agent.prev_move)):
                    ties.append((factor,'w'))
            elif factor[0]==(agent.posX,agent.posY+1):
                if (agent.prev_move == None) or (agent.prev_move != None and getOppositeDirection('s') != agent.prev_move):
                    ties.append((factor,'s'))
            elif factor[0]==(agent.posX,agent.posY-1):
                if (agent.prev_move == None) or (agent.prev_move != None and getOppositeDirection('n') != agent.prev_move):
                    ties.append((factor,'n'))
            elif factor[0]==(agent.posX,agent.posY):
                ties.append((factor,'q'))

    if len(ties) == 0:
        for factor in Qfactors:
            if factor[1] == minQ:
                if factor[0]==(agent.posX+1,agent.posY):
                    ties.append((factor,'e'))
                elif factor[0]==(agent.posX-1,agent.posY):
                    ties.append((factor,'w'))
                elif factor[0]==(agent.posX,agent.posY+1):
                    ties.append((factor,'n'))
                elif factor[0]==(agent.posX,agent.posY-1):
                    ties.append((factor,'s'))
                elif factor[0]==(agent.posX,agent.posY):
                    ties.append(factor,'q')

    bestMove = ties[0][0][0]

    if bestMove==(agent.posX+1,agent.posY):
        ret= 'e'
    elif bestMove==(agent.posX-1,agent.posY):
        ret= 'w'
    elif bestMove==(agent.posX,agent.posY+1):
        ret= 'n'
    elif bestMove==(agent.posX,agent.posY-1):
        ret= 's'
    elif bestMove==(agent.posX,agent.posY):
        ret= 'q'
    return ret,minCost
"""

def multiAgentRollout(networkVertices, networkEdges, networkAgents, taskPos, agent):
    # Change to robotarium coordinates
    for vertex in networkVertices:
        try:
            assert (vertex, vertex) in networkEdges
        except AssertionError:
            networkEdges.append((vertex,vertex))

    currentPos = networkAgents.copy()
    currentTasks = taskPos.copy()

    assert len(currentTasks) > 0
    minCost = float('inf')
    bestMove = None
    Qfactors = []
    prevCost = 0

    agent_ID = list(agent.keys())[0]
    # print("Cluster ID: ", agents[agent_ID-1].clusterID)
    agent_pos = agent[agent_ID]

    for e in networkEdges:
        assert e[0] in networkVertices
        assert e[1] in networkVertices
        if e[0]==agent_pos:
            if '3' in verbose or verbose == '-1' or 'c' in verbose:
                print("EOI: ", e)
            tempCurrentTasks = currentTasks.copy()
            tempPositions = currentPos.copy()

            if e[1] != e[0]:
                cost = prevCost+1
            else:
                cost = prevCost+0.1

            tempPositions[agent_ID] = e[1]
            if tempPositions[agent_ID] in tempCurrentTasks:
                tempCurrentTasks.remove(tempPositions[agent_ID])
            
            flag = False
            for a_ID in tempPositions:
                if len(tempCurrentTasks) == 0:
                    break
                if flag == True:
                    dist,path = bfsNearestTask(networkVertices,networkEdges,
                                                    tempPositions[a_ID],
                                                    tempCurrentTasks)
                    assert dist != None
                    tempPositions[a_ID] = path[1]
                    cost += 1

                    if tempPositions[a_ID] in tempCurrentTasks:
                        tempCurrentTasks.remove(tempPositions[a_ID])
                if flag == False:
                    flag = True if (a_ID == agent_ID) else False
            if '3' in verbose or verbose == '-1' or 'c' in verbose:
                print("Task List: ", tempCurrentTasks)

            while len(tempCurrentTasks) > 0:
                if '3' in verbose or verbose == '-1':
                    print("\tRemaining Tasks: ", len(tempCurrentTasks))
                for a_ID in networkAgents:
                    shortestDist = float('inf')
                    bestNewPos = None
                    assert tempPositions[a_ID] in networkVertices
                    dist, path = bfsNearestTask(networkVertices, networkEdges, tempPositions[a_ID], tempCurrentTasks)
                    
                    if dist == None:
                        """
                            should only arise if an agent is alone 
                            in a connected component
                        """
                        bestNewPos = tempPositions[a_ID]
                    else:
                        bestNewPos = path[1]
                    if '3' in verbose or verbose == '-1':
                        print(f"\tAgent {tempPositions[a_ID]} moves " +
                        f"to {bestNewPos}")
                    assert bestNewPos != None
                    tempPositions[a_ID] = bestNewPos
                    cost += 1
                    if tempPositions[a_ID] in tempCurrentTasks:
                        tempCurrentTasks.remove(tempPositions[a_ID])
                    if len(tempCurrentTasks) == 0:
                        break
            if '3' in verbose or verbose == '-1':
                print("\tCost-to-go for EOI: ", cost)

            if cost < minCost:
                minCost = cost
                bestMove = e[1]

            Qfactors.append((e[1], cost))
            del tempPositions
            del tempCurrentTasks

    assert bestMove != None     ## should at least wait... 
    print("Agent: ", agent_ID)
    print("Qfactors: ", Qfactors)
    minQ = float('inf')
    for factor in Qfactors:
        if factor[1] < minQ:
            minQ = factor[1]

    """
    Note: 
    ---------------------------

    Not asserting that agents not move back to previous positions
    as we know that there can be oscillations of larger periods and
    we can't check all possible periods. 

    Oscillations should implicitly not occur. 

    """

    ## collect ties in Qfactor values
    ties = []
    for factor in Qfactors:
        if factor[1] == minQ:
            ties.append(factor)

    assert len(ties) >= 1 ## some move must get finite cost
    if '3' in verbose or verbose == '-1' or 'c' in verbose:
        print(agent_ID, agents[agent_ID-1].posX, agents[agent_ID-1].posY
            , ties)

    """
    Question: 
    ---------------------------

    If we don't break ties randomly, then why bother creating a
    list of ties and selecting the i-th  Qfactor (for some i)? 

    """

    """
    Note: 
    ---------------------------

    Breaking ties by picking the wait-move if possible, 
    otherwise pick the last move in the ties list. 

    """

    """
    TO-DO: 
    --------------------------- 
    
    Use Euclidean distance to closest task and choose action
    that makes progress towards this task. 

    """
    for factor in ties:
        bestMove = factor[0]
        if factor[0] == agent_pos:
            break

    if bestMove == (round(agent_pos[0]+step, 1),agent_pos[1]):
        ret = 'e'
    elif bestMove == (round(agent_pos[0]-step, 1),agent_pos[1]):
        ret = 'w'
    elif bestMove == (agent_pos[0],round(agent_pos[1]+step, 1)):
        ret = 'n'
    elif bestMove == (agent_pos[0],round(agent_pos[1]-step, 1)):
        ret = 's'
    elif bestMove == agent_pos:
        ret = 'q'

    if '3' in verbose or verbose == '-1' or 'c' in verbose:
        print(ties)
        print("Move choic: ", ret)
        print()
    return ret, minCost

#Only centroid start this
def clusterMultiAgentRollout(centroidID, networkVertices, networkEdges, networkAgents, taskPos, agent):
    # Change to robotarium coordinates
    (x_offset, y_offset) = agent[centroidID]
    agentPositions = {}
    for a_ID in networkAgents:
        agentPositions[a_ID] = (round(agents[a_ID-1].posX-networkAgents[a_ID][0], 1),
                                round(agents[a_ID-1].posY-networkAgents[a_ID][1], 1))
    tempTasks = taskPos.copy()

    allPrevMoves = {}
    for a_ID in networkAgents:
        allPrevMoves[a_ID] = []

    while len(tempTasks) > 0:
        waitAgents = []
        prevMoves = {}
        for a_ID in networkAgents:
            agent_pos = agentPositions[a_ID]
            assert agent_pos in networkVertices
            if only_base_policy:
                ## need to move to global coordinates
                # taskList = [(task[0]+x_offset,task[1]+y_offset) for task \
                #             in tempTasks]
                # global_temp_pos = (agentPositions[a_ID][0]+x_offset,
                #                     agentPositions[a_ID][1]+y_offset)
                move,c = getClosestClusterTask(agent_pos=agentPositions[a_ID], 
                                                taskList=tempTasks, 
                                                lookupTable=None,
                                                vertices=networkVertices,
                                                edges=networkEdges)
            else:
                move,c = multiAgentRollout(networkVertices, networkEdges,
                                        agentPositions, tempTasks, 
                                        {a_ID:agent_pos})

            ## NOTE: Not needed - handled in stateUpdate
            ## Check if the move is causing a collision and if so then instead append a wait move
            # collision = False
            # if move == 'n':
            #     temp_pos = (agent_pos[0],round(agent_pos[1]+step, 1))
            # elif move == 's':
            #     temp_pos = (agent_pos[0],round(agent_pos[1]-step, 1))
            # elif move == 'e':
            #     temp_pos = (round(agent_pos[0]+step, 1),agent_pos[1])
            # elif move == 'w':
            #     temp_pos = (round(agent_pos[0]-step, 1),agent_pos[1])
            # for a_id in prevMoves:
            #     if temp_pos == agentPositions[a_id]:
            #         print("Agent: ", a_ID, " colliding with Agent: ", a_id)
            #         move = 'q'
            #         collision = True
            #         if a_id in waitAgents:
            #             waitAgents.append(a_ID)
            #         break
            
            prevMoves[a_ID] = move
            allPrevMoves[a_ID].append(move)
            if move == 'n':
                agentPositions[a_ID] = (agent_pos[0],round(agent_pos[1]+step, 1))
            elif move == 's':
                agentPositions[a_ID] = (agent_pos[0],round(agent_pos[1]-step, 1))
            elif move == 'e':
                agentPositions[a_ID] = (round(agent_pos[0]+step, 1),agent_pos[1])
            elif move == 'w':
                agentPositions[a_ID] = (round(agent_pos[0]-step, 1),agent_pos[1])
            elif move == 'q':
                waitAgents.append(a_ID)
                pass

            if agentPositions[a_ID] in tempTasks:
                tempTasks.remove(agentPositions[a_ID])

            if len(tempTasks) == 0:
                break

    longest = 0
    for a_ID in allPrevMoves:
        if len(allPrevMoves[a_ID]) > longest:
            longest = len(allPrevMoves[a_ID])

    for a_ID in allPrevMoves:
        while len(allPrevMoves[a_ID]) < longest:
            allPrevMoves[a_ID].append('q')

    if '3' in verbose or verbose == '-1':
        print("allPrevMoves: ", allPrevMoves)

    """
    If we are using depots to return to, then we need to 
    route all agents back to the original position of the 
    leader agent. 

    Note that the original position of the leader is simply
    the tuple (0,0) in the local view. 

    """
    """
    if depots:
        for a_ID in agentPositions:
            while agentPositions[a_ID] != (0,0):
                dist,path = ut.dirShortestPath(networkVertices, networkEdges,
                                            agentPositions[a_ID], 
                                            (0,0))
                assert dist != None 
                newPos = path[1]
                if newPos == (agentPositions[a_ID][0],agentPositions[a_ID][1]+1):
                    allPrevMoves[a_ID].append('n')
                    agentPositions[a_ID] = (agentPositions[a_ID][0],
                                            agentPositions[a_ID][1]+1)
                elif newPos == (agentPositions[a_ID][0],agentPositions[a_ID][1]-1):
                    allPrevMoves[a_ID].append('s')
                    agentPositions[a_ID] = (agentPositions[a_ID][0],
                                            agentPositions[a_ID][1]-1)
                elif newPos == (agentPositions[a_ID][0]+1,agentPositions[a_ID][1]):
                    allPrevMoves[a_ID].append('e')
                    agentPositions[a_ID] = (agentPositions[a_ID][0]+1,
                                            agentPositions[a_ID][1])
                elif newPos == (agentPositions[a_ID][0]-1,agentPositions[a_ID][1]):
                    allPrevMoves[a_ID].append('w')
                    agentPositions[a_ID] = (agentPositions[a_ID][0]-1,
                                            agentPositions[a_ID][1])
                else: 
                    raise ValueError("Agent needs to move.")
    """
    return allPrevMoves

def mergeTimelines():
    # #print("Merging... ")
    # Naive Merge
    for agent in agents:
        agent.posX = agent.posX_prime
        agent.posY = agent.posY_prime
        agent.color = agent.color_prime
        agent.cost = agent.cost_prime

        agent.dir=agent.dir_prime
        agent.clusterID=(agent.clusterID_prime.copy())
        agent.parent=(agent.parent_prime)
        agent.children=agent.children_prime
        agent.stateMem=(agent.stateMem_prime)
        agent.viewEdges=(agent.viewEdges_prime)
        agent.viewVertices=(agent.viewVertices_prime)
        agent.viewAgents=(agent.viewAgents_prime)
        agent.viewTasks=(agent.viewTasks_prime)
        agent.eta=agent.eta_prime
        agent.message=agent.message_prime
        agent.clusterVertices=(agent.clusterVertices_prime.copy())
        agent.clusterEdges=(agent.clusterEdges_prime)
        agent.clusterTasks=(agent.clusterTasks_prime)
        agent.clusterAgents=(agent.clusterAgents_prime)
        agent.localVertices=(agent.localVertices_prime)
        agent.localEdges=(agent.localEdges_prime)
        agent.localTasks=(agent.localTasks_prime)
        agent.localAgents=(agent.localAgents_prime)
        agent.childMarkers=(agent.childMarkers_prime)
        agent.xOffset=agent.xOffset_prime
        agent.yOffset=agent.yOffset_prime
        agent.resetCounter=agent.resetCounter_prime
        agent.moveList=(agent.moveList_prime)
        agent.marker=agent.marker_prime
        agent.dfsNext=agent.dfsNext_prime

        """
        if agent.gui_split == True:
            #print(agent.ID, agent.color)
            if visualizer:
                changeCell(agent.posX, agent.posY, "agent", agent.color)
            agent.gui_split = False
        """

    ## Merge Mutual Connections
    for agent in agents:
        ## if x is not the parent of agent, then remove x from agent.children
        for x in agents:
            if x != agent and x != agent.parent:
                if agent in x.children:
                    x.children.remove(agent)

def getClosestClusterTask(agent_pos, taskList, lookupTable, **kwargs):
    (agent_posX, agent_posY) = agent_pos
    min_dist = float('inf')
    best_move = None
    if agent_posX == 0:
        pos_x = 0.0
    else:
        pos_x = agent_posX
    if agent_posY == 0:
        pos_y = 0.0
    else:
        pos_y = agent_posY
    for task in taskList:
        task_co = (round(task[0],1),round(task[1],1))
        if lookupTable != None:
            try:
                dist, path = lookupTable[str((pos_x, pos_y))][str(task_co)]
            except (AssertionError, KeyError):
                continue
        else:
            dist, path = dirShortestPath(kwargs['vertices'], 
                                            kwargs['edges'],
                                            (pos_x, pos_y), task_co)
        if dist != None and dist < min_dist:
            min_dist = dist
            best_move = path[1]
    
    if best_move == (round(agent_posX+step,1), agent_posY):
        next_dir = 'e'
    elif best_move == (round(agent_posX-step,1), agent_posY):
        next_dir = 'w'
    elif best_move == (agent_posX, round(agent_posY-step,1)):
        next_dir = 's'
    elif best_move == (agent_posX, round(agent_posY+step,1)):
        next_dir = 'n'
    elif best_move == None:
        next_dir = 'q'
    else:
        raise ValueError("Best Move is not a move. ")

    return next_dir, min_dist

def main():
    totalCost = 0
    waitCost = 0
    df = pd.DataFrame({'Centralized':[], 'Seed #': [], 'Exp Seed #': [],
                        'Rows': [], 'Cols': [], 'Wall Prob': [],
                        '# of Agents': [], '# of Tasks': [],
                        'k': [], 'psi': [], 'Total Time (s)': [],
                        '# of Exploration': []})
    # lookupTable = offlineTrainRes

    if centralized:
        begin = time.time()
        """
        while len(taskVertices) > 0:
            moves = {}
            for a in agents:
                if only_base_policy:
                    move, c = getClosestClusterTask(a, taskVertices,
                                                    lookupTable)
                else:
                    move, c = multiAgentRolloutCent(vertices, adjList, agents,
                                                    taskVertices, a, moves,
                                                    lookupTable)
                moves[a] = move
                a.move(move)

                if move != 'q':
                    totalCost += 1
                else:
                    waitCost += 1
            stateUpdate()

            time.sleep(wait_time)
            sys.stdout.flush()
            # if visualizer:
            #     costLabel = Label(root, text='Total Cost: ' + str(totalCost))
            #     costLabel.grid(row=rows+1, column=cols-3,columnspan=4)
        end = time.time()
        totalTime = end-begin
        print("Done. ")
        new_data['# of Exploration Steps'] = str(0)
        new_data['Wait Cost'] = str(waitCost)
        # if visualizer:
        #     quitButton.invoke()
        """
    else:
        try:
            begin =  time.time()
            explore_steps = 0
            r_pos = r_env.get_poses()

            # robot_markers = [r_env.axes.scatter(r_pos[0,ii], r_pos[1,ii], s=marker_size_robot, marker='s', facecolors='none',edgecolors='none',linewidth=7) for ii in range(A)]

            r_env.step()
            rounds = 0
            COMPLETION_PARAM = 0.0
            target_completion = int(COMPLETION_PARAM * len(taskVertices))
            cluster_count = 0.0
            while len(taskVertices) > target_completion:
                # time.sleep(wait_time)

                rounds += 1

                for a in agents:
                    a.updateView(r_pos)

                updateAgentToAgentView(r_pos)
                sys.stdout.flush()

                """
                ---------------------------- SOAC ----------------------------
                """
                ## Phase 1
                for a in agents:
                    a.eta += len(a.viewTasks)
                    a.eta_prime += len(a.viewTasks_prime)

                """
                Begin Phase 2: 
                ----------------------------

                For each agent a in agents that can see a task, search a's view for another agent that is already part of a cluster.
                    If no such agent in a's view, make a the centroid.

                """
                for a in agents:
                    # #print(a.viewTasks)
                    if a.eta > 0: ## a can see a task...
                        flag = True ## assume a is going to be the centroid of a new cluster...

                        for x in a.viewAgents:
                            if len(x.clusterID) > 0 and x != a: ## a sees agent x that already belongs to a cluster...
                                flag = False ## a cannot be the centroid of a new cluster...

                        if flag: ## a is the new centroid...
                            a.clusterID_prime.append(a.ID)
                            a.parent_prime = None ## a has no parent since it is the centroid...

                mergeTimelines()

                ### Begin Phase RR (Round Robin):
                """
                Observe that at this stage the only agents in a cluster are the ones that can see at least one task.

                    For each agent a in agents that can see a task, check its view for other agents that can also see a task
                    and perform a round robin among them to pick a single centroid. This is done by performing "leader election"
                    by picking the agent that has the highest agent ID.

                """
                # #print("RR:")
                for a in agents:
                    if len(a.clusterID) > 0:
                        for x in a.viewAgents:
                            if len(x.clusterID) > 0  and x != a:
                                if x.ID > a.ID:
                                    a.clusterID_prime = []
                                    a.children_prime = []
                                    a.parent_prime = None
                mergeTimelines()
                if '1' in verbose or verbose == '-1':
                        print("Phase RR: ")
                        for a in agents:
                            print(a.ID, a.posX, a.posY, 
                                a.color, a.clusterID, end=" ")
                            if a.parent != None:
                                print(a.parent.ID)
                            else:
                                print()
                            print("\tChildren: ", end=" ")
                            for b in a.children:
                                print(b.ID, end=" ")
                            print()
                        print()

                ## Update colors for centroids...
                for a in agents:
                    if len(a.clusterID) > 0 and a.ID == a.clusterID[0]:
                        a.color_prime = colors.pop(0)
                        a.gui_split = True

                mergeTimelines()

                for i in range(psi):
                    """
                    Begin Phase 3:
                    ----------------------------

                    For each agent a that does not belong to any cluster, 
                    search its view for agents that are already in a cluster
                    and add a to be their children/join their cluster.

                    """
                    for a in agents:
                        if len(a.clusterID) == 0:
                            for x in a.viewAgents:
                                if len(x.clusterID) > 0 and  x != a:
                                    a.clusterID_prime.append(x.clusterID[0])
                                    a.parent_prime = x
                                    a.color_prime = x.color
                                    a.gui_split = True

                                    if a not in x.children and a != x:
                                        x.children_prime.append(a)

                    mergeTimelines()
                    if '1' in verbose or verbose == '-1':
                        print("Phase 3: ")
                        for a in agents:
                            print(a.ID, a.posX, a.posY, 
                                a.color, a.clusterID, end=" ")
                            if a.parent != None:
                                print(a.parent.ID)
                            else:
                                print()
                            print("\tChildren: ", end=" ")
                            for b in a.children:
                                print(b.ID, end=" ")
                            print()
                        print()

                    """
                    Begin Phase 4:
                    ----------------------------

                    If an agent a is part of more than one cluster, then it
                    becomes a centroid and forms a new Super Cluster that
                    contains all agents in both clusters.

                    """
                    for a in agents:
                        ## delete duplicates...
                        a.clusterID = list(set(a.clusterID))
                        if len(a.clusterID) > 1:
                            a.clusterID_prime = [a.ID] ## become the new centroid...
                            a.message_prime = True
                            a.parent_prime = None ## no parent for centroids...
                            a.color_prime = colors.pop(0)
                            a.gui_split = True

                            for x in a.viewAgents:
                                if x.message == False and x != a:
                                    x.clusterID_prime = [a.ID]
                                    x.message_prime = True ## now x has received the message about the new centroid...
                                    x.parent_prime = a ## a is the new parent...
                                    x.color_prime = a.color ## get a's color...
                                    x.gui_split = True

                                    if x not in a.children:
                                        a.children_prime.append(x)
                                    x.children_prime = []

                    mergeTimelines()
                    if '1' in verbose or verbose == '-1':
                        print("Phase 4: ")
                        for a in agents:
                            print(a.ID, a.posX, a.posY, 
                                a.color, a.clusterID, end=" ")
                            if a.parent != None:
                                print(a.parent.ID)
                            else:
                                print()
                            print("\tChildren: ", end=" ")
                            for b in a.children:
                                print(b.ID, end=" ")
                            print()
                        print()                    

                    """
                    Begin Phase 4.5:
                    ----------------------------

                    Maintain consistency among parent-child relationships.

                    """
                    for i in range(2,N):
                        for a in agents:
                            if len(a.clusterID) > 0:
                                if a.parent != None:
                                    a.clusterID_prime = a.parent.clusterID
                                    a.color_prime = a.parent.color
                                    if a not in a.parent.children:
                                        a.parent.children_prime.append(a)

                        mergeTimelines()
                    ## perform a final visualizer update
                    for a in agents:
                        if len(a.clusterID) > 0:
                            a.gui_split = True
                    mergeTimelines()
                    if '1' in verbose or verbose == '-1':
                        print("Phase 4.5: ")
                        for a in agents:
                            print(a.ID, a.posX, a.posY, 
                                a.color, a.clusterID, end=" ")
                            if a.parent != None:
                                print(a.parent.ID)
                            else:
                                print()
                            print("\tChildren: ", end=" ")
                            for b in a.children:
                                print(b.ID, end=" ")
                            print()
                        print()


                    for i in range(2,N):
                        """
                        Begin Phase 5:
                        ----------------------------

                        Transfer the message that a new cluster has been 
                        formed to all children of agents within the super
                        cluster's centroid view using the message flag.

                        """
                        for a in agents:
                            if a.message == True:
                                for x in a.viewAgents:
                                    if x.message == False and x != a:
                                        x.clusterID_prime = a.clusterID.copy()
                                        x.message_prime = True
                                        x.parent_prime = a
                                        x.color_prime = a.color
                                        x.gui_split = True

                                        if x not in a.children:
                                            a.children_prime.append(x)
                                        x.chilren_prime = []

                        mergeTimelines()

                if '1' in verbose or verbose == '-1':
                    print("Phase 5: ")
                    for a in agents:
                        print(a.ID, a.posX, a.posY, 
                            a.color, a.clusterID, end=" ")
                        if a.parent != None:
                            print(a.parent.ID)
                        else:
                            print()
                        print("\tChildren: ", end=" ")
                        for b in a.children:
                            print(b.ID, end=" ")
                        print()
                    print()

                """
                Begin Phase 6:
                ----------------------------

                Reset message flag for all agents

                """
                for a in agents:
                    a.message = False

                """
                Begin Phase 7: 
                ----------------------------

                Remove any stray children. 

                """
                for a in agents:
                    if a.parent != None and len(a.clusterID) > 0:
                        for b in a.viewAgents:
                            if b != a and b != a.parent: 
                                if a in b.children: 
                                    b.children.remove(a)

                # time.sleep(wait_time)

                if '1' in verbose or verbose == '-1':
                    print("Phase 7: ")
                    for a in agents:
                        print(a.ID, a.posX, a.posY, 
                            a.color, a.clusterID, end=" ")
                        if a.parent != None:
                            print(a.parent.ID)
                        else:
                            print()
                        print("Children: ", end=" ")
                        for b in a.children:
                            print(b.ID, end=" ")
                        print()
                    print()

                """
                Begin Phase 8: 
                ----------------------------

                !!! This phase should not be necessary if everything 
                    above works correctly. (Right?)

                Assert that all colors match to clusterID number. 

                """
                for _ in range(N):
                    for a in agents:
                        if a.parent != None:
                            a.color_prime = a.parent.color
                            if a.color != a.parent.color:
                                ## only update visualizer if color changes
                                a.gui_split = True
                    mergeTimelines()
                
                ## Commented for bulk trials
                # r_pos = r_env.get_poses()
                # for i in range(A):
                #     agent = agents[i]
                #     clusterM = 0
                #     if len(agent.getCluster()) > 0:
                #         clusterM = agent.getCluster()[0]
                #     robot_markers[i] = r_env.axes.scatter(r_pos[0,i], r_pos[1,i], s=marker_size_robot, marker=marker_shapes[clusterM], facecolors='none',edgecolors=colorIndex[agent.getColor()-1],linewidth=7)
                # r_env.step()

                if '1' in verbose or verbose == '-1':
                    print("End of SOAC... ")
                    for a in agents:
                        print(a.ID, a.posX, a.posY, 
                            a.color, a.clusterID, end=" ")
                        if a.parent != None:
                            print(a.parent.ID)
                        else:
                            print()
                        print("\tChildren: ", end=" ")
                        for b in a.children:
                            print(b.ID, end=" ")
                        print()
                    print()

                unique_clusters = set()
                for a in agents:
                    if a.parent == None and len(a.clusterID) != 0:
                        unique_clusters.add(a.clusterID[0])
                # cluster_count = ((cluster_count*(rounds-1))+ \
                #                 len(unique_clusters))/rounds
                cluster_count += len(unique_clusters)

                """
                ---------------------------- LMA ----------------------------
                """
                if '2' in verbose or verbose == '-1':
                    print("Beginning LMA... ")

                for a in agents:
                    a.message = False # turn off message flag

                """
                Begin Phase 0.5: 
                ----------------------------

                Offset passing within a cluster before map sharing.

                """
                for a in agents:
                    if len(a.clusterID) != 0 and a.parent == None:
                        a.xOffset = a.posX
                        a.yOffset = a.posY
                        a.message = True
                for _ in range(N):
                    for a in agents:
                        if a.message == True:
                            for b in a.children:
                                if b.message == False: 
                                    b.xOffset = a.xOffset
                                    b.yOffset = a.yOffset
                                    b.message = True
                for a in agents:
                    a.message = False

                """
                Begin Phase 1: 
                ----------------------------

                Obtain mapOffset values and adjust local view for 
                leader agents. Begin map sharing with children of
                all leader agents. 

                """
                for a in agents:
                    if a.parent==None and len(a.clusterID)>0:
                        for b in a.children:
                            a.childMarkers.append(0)

                        a.updateLocalView()
                        a.clusterAgents=a.localAgents.copy()
                        ## get rid of agents that are not part of any cluster...
                        to_del = []
                        for a_ID in a.clusterAgents:
                            if agents[a_ID-1].clusterID == []:
                                to_del.append(a_ID)
                        for a_ID in to_del:
                            del a.clusterAgents[a_ID]

                        s=a.clusterAgents.copy()
                        # get rid of agents that are not in the same cluster...
                        for a_ID in s:
                            if agents[a_ID-1].clusterID[0]!=a.clusterID[0]:
                                del a.clusterAgents[a_ID]

                        a.clusterVertices=a.localVertices.copy()
                        a.clusterEdges=a.localEdges.copy()
                        a.clusterTasks=a.localTasks.copy()

                        for b in a.children:
                            b.clusterVertices=b.clusterVertices.union(a.clusterVertices)
                            b.clusterEdges=b.clusterEdges.union(a.clusterEdges)
                            b.clusterTasks=b.clusterTasks.union(a.clusterTasks)
                            b.clusterAgents={**b.clusterAgents, 
                                                **a.clusterAgents}
                        a.message=True

                """
                Begin Phase 2: 
                ----------------------------

                Map sharing with all agents in the cluster. 

                """
                for i in range(N):
                    for a in agents:
                        if a.parent != None and a.parent.message:
                            for b in a.children:
                                a.childMarkers.append(0)

                            l=a.mapOffset(a.parent.xOffset,a.parent.yOffset,
                                          a.viewVertices,
                                          a.viewEdges,
                                          a.viewTasks,
                                          a.viewAgents)

                            for x in l[3]:
                                flag=False
                                for y in a.clusterAgents:
                                    if x == y:
                                        flag=True
                                if not flag:
                                    a.clusterAgents[x] = l[3][x]
                            s=a.clusterAgents.copy()

                            for b in s:
                                if len(agents[b-1].clusterID)==0:
                                    del a.clusterAgents[b]
                                elif agents[b-1].clusterID[0]!=a.clusterID[0]:
                                    del a.clusterAgents[b]

                            a.clusterVertices=a.clusterVertices.union(l[0])
                            a.clusterEdges=a.clusterEdges.union(l[1])
                            a.clusterTasks=a.clusterTasks.union(l[2])

                            for b in a.children:
                                a.childMarkers.append(0)
                                b.clusterVertices=b.clusterVertices.union(a.clusterVertices)
                                b.clusterEdges=b.clusterEdges.union(a.clusterEdges)
                                b.clusterTasks=b.clusterTasks.union(a.clusterTasks)
                                b.clusterAgents={**b.clusterAgents, 
                                                    **(a.clusterAgents)}

                            a.message=True
                            a.parent.resetCounter+=1
                            if a.parent.resetCounter==len(a.parent.children):
                                a.parent.message=False

                """
                Begin Phase 3: 
                ----------------------------

                Propogate map information back to leader. 

                """
                for i in range(N):
                    for a in agents:
                        if len(a.clusterID)>0 and a.parent!= None:
                            a.parent.clusterVertices=a.parent.clusterVertices.union(a.clusterVertices)
                            a.parent.clusterEdges=a.parent.clusterEdges.union(a.clusterEdges)
                            a.parent.clusterTasks=a.parent.clusterTasks.union(a.clusterTasks)
                            a.parent.clusterAgents={**a.parent.clusterAgents, 
                                                    **(a.clusterAgents)}

                ## Known error check
                for agent in agents:
                    agent_pos = {}
                    for a in agent.clusterAgents:
                        agent_pos[a]=(round(agents[a-1].posX-agent.clusterAgents[a][0],1),round(agents[a-1].posY-agent.clusterAgents[a][1],1))
                        cluster = agents[a-1].clusterID[0]
                    try:
                        assert len(set(agent_pos.values()).intersection(set(agent.clusterTasks))) == 0
                    except AssertionError:
                        task_list = list(agent.clusterTasks)
                        task_list = [(agents[cluster-1].posX+task[0], 
                                    agents[cluster-1].posY+task[1]) for task in task_list]
                        if '2' in verbose or verbose == '-1':
                            print(agent_pos, agent.clusterTasks, 
                                    set(agent_pos.values()).intersection(set(agent.clusterTasks)), set(task_list).intersection(set(taskVertices)))
                        raise AssertionError

                """
                ---------------------------- T-MAR ----------------------------
                """
                if '3' in verbose or verbose == '-1':
                    print("Beginning T-MAR... ")
                clusterMoves = {}
                for a in agents:
                    if a.parent==None and len(a.clusterID)>0:
                        agent = {a.ID: a.clusterAgents[a.ID]}
                        clusterMoves[a.ID] = clusterMultiAgentRollout(a.ID, 
                                                            list(a.clusterVertices),
                                                            list(a.clusterEdges),
                                                            a.clusterAgents,
                                                            list(a.clusterTasks),
                                                            agent)

                ## pass this moves list to all agents in agent tree
                if '3' in verbose or verbose == '-1':
                    print("Done.", clusterMoves)
                ## Pass this information to all children
                for a in agents:
                    if a.parent == None and len(a.clusterID) > 0:
                        a.message = True
                        a.moves = clusterMoves[a.ID][a.ID]
                    else:
                        a.message = False
                for i in range(N):
                    ## centroid shares this information with its children
                    for a in agents:
                        if a.message == True:
                            for b in a.children:
                                if b.message == False:
                                    ## receive move information from parent
                                    b.moves = clusterMoves[b.clusterID[0]][b.ID]
                                    b.message = True

                ## loose upper bound... 
                # num_iterations = int(np.pi*5*(k*(2**(psi+1)-1))**2)
                # Sync execution of actions all at once
                num_iterations = (2**(psi))*k
                tempTasks = taskVertices.copy()
                clusterAgents = [x for x in agents if len(x.clusterID) > 0]
                iters = 0
                while len([x for x in clusterAgents if len(x.moves) > 0]) > 0 if len(clusterAgents) > 0 else iters < num_iterations:
                    iters += 1
                    if '3' in verbose or verbose == '-1':
                        print("=======Round: {}/{}".format(iters,num_iterations))
                        print(len(taskVertices), totalCost)
                    if 't' in verbose or verbose == '-1':
                        print("Remaining Tasks: ", len(taskVertices))
                    if len(taskVertices) == 0:
                        break
                    ## Still exploring
                    for a in agents:
                        if a.exp_dist_remaining != 0:
                            a.clusterID = []

                    ## Let this be done in tandem with the stateUpdate with all the collision resolution
                    goal_points, totalCost, waitCost, explore_steps = stateUpdate(r_pos, totalCost, waitCost, explore_steps, tempTasks)

                    # print(r_pos)
                    # print(goal_points)

                    ### Execute Sync position updates
                    while (np.size(at_pose(r_pos, goal_points)) != A):
                        # Get poses of agents
                        r_pos = r_env.get_poses()

                        # Comment this for bulk runs
                        # for i in range(A):
                        #     agent = agents[i]
                            # robot_markers[i].set_offsets(r_pos[:2,i].T)

                        # Create unicycle control inputs
                        dxu = unicycle_pose_controller(r_pos, goal_points)
                        ## for position only controller use
                        # dxu = unicycle_pose_controller(r_pos, goal_points[:2][:])
                        # Create safe control inputs (i.e., no collisions)
                        dxu = uni_barrier_cert(dxu, r_pos)
                        # Set the velocities
                        r_env.set_velocities(np.arange(A), dxu)
                        # Iterate the simulation
                        r_env.step()
                    # print("Finished 1 Step of robotarium move")
                    # time.sleep(wait_time)
                    r_pos = r_env.get_poses()
                    ## Remove Tasks
                    remove_t = set()
                    for t in taskVertices:
                        if any(np.linalg.norm(r_pos[:2,:] - np.reshape(np.array(t), (2,1)), axis=0) < 0.1):
                            remove_t.add(t)
                            # Comment this for bulk runs
                            # for ts in taskss:
                            #     if ts.get_offsets()[0][0] == t[0] and ts.get_offsets()[0][1] == t[1]:
                            #         ts.set_visible(False)
                            #         break
                    for t in remove_t:
                        taskVertices.remove(t)
                    r_env.step()

                ## Check if there are agents which are still waiting for their trajectory to complete
                for a in agents:
                    if len(a.clusterID) > 0 and len(a.moves) > 0:
                        print("Agent: ", a.ID, " has still ", len(a.moves), " moves to complete")

                for a in agents:
                    a.deallocate()

                    a.reset()
                    a.resetColor()

                    a.posX_prime = a.posX
                    a.posY_prime = a.posY

                # time.sleep(wait_time)
                # r_pos = r_env.get_poses()
                # for i in range(A):
                #     agent = agents[i]
                #     robot_markers[i].set_edgecolors(colorIndex[agent.getColor()-1])
                # r_env.step()
                sys.stdout.flush()
                # if visualizer:
                #     costLabel=Label(root,text='Total Cost: '+str(totalCost))
                #     costLabel.grid(row=rows+1,column=cols-3,columnspan=4)

            end = time.time()
            #print("Decentralized Cost: ", totalCost)

            totalTime = end-begin
            #print("Decentralized Time: ", end - begin)

            new_data['# of Exploration Steps'] = explore_steps
            new_data['Wait Cost'] = waitCost
            # if visualizer:
            #     quitButton.invoke()

            r_env.call_at_scripts_end()

        except KeyboardInterrupt:
            sys.exit(0)

    new_data['Total Cost'] = str(totalCost)
    new_data['Total Time (s)'] = str(totalTime)
    new_data['Average Cluster Count'] = str(cluster_count)
    if verbose == '-1':
        print(new_data)
    print(f"Total Cost: {totalCost}; Total Time (s): {totalTime};" + \
     f" Wait Cost: {waitCost}; Exploration Cost: {explore_steps}")

    # df = df.append(new_data, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    # create unique filename...
    filename = "./results/exps_" + \
        str(seed) + "_" + str(numTasks) + "_" + \
        str(k) + "_exp_seed_"+ str(exp_seed) +"_base_" + str(only_base_policy) + ".xlsx"

    if os.path.exists(filename) == False:
        with pd.ExcelWriter(filename, mode="w") as writer:
            df.to_excel(writer, index=False)
            print("Creating file: ", filename)
    else:
        print("File already exists... ")

#Driver
def start():
    #print('starting')
    t=threading.Thread(target=main)
    t.start()

# if __name__ == "__main__":
#   main()
#Add Interface
"""
if visualizer:
    goButton=Button(root,text='Start',pady=10,padx=50,command=start)
    goButton.grid(row=rows+1, column=0, columnspan=cols-4)
    quitButton=Button(root, text="Quit", command=root.destroy)
    quitButton.grid(row=rows+2, column=0, columnspan=cols-4)
    costLabel=Label(root,text='Total Cost: '+str(totalCost))
    costLabel.grid(row=rows+1,column=cols-3,columnspan=4)

    goButton.invoke()

    root.mainloop()
else:
"""
main()

