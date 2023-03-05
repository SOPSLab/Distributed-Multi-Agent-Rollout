from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import pandas as pd
import random
import time
import itertools
import copy
import openpyxl

import utils as ut
from init import getParameters

wait_time = 0

totalCost = 0
padding = 20

rows, cols, A, numTasks, k, psi, centralized, visualizer, wall_prob, seed, collisions, exp_strat, only_base_policy = getParameters()

new_data = {'Centralized':str(centralized), 'Seed #': str(seed),
            'Rows': str(rows), 'Cols': str(cols), 'Wall Prob': str(wall_prob),
            '# of Agents': str(A), '# of Tasks': str(numTasks), 'k': str(k),
            'psi': str(psi), 'Only Base Policy': str(only_base_policy)}

if visualizer:
    root = Tk()
    root.resizable(height=None, width=None)
    root.title("RL Demo")

gridLabels=[]
holeProb=0.25
pauseFlag=False
memSize=10

agentImages=[]
edgeList=[]
vertices=[]
global colors
colors=[]
for i in range(100):
    colors=colors+[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

if visualizer:
    #agent images
    for i in range(31):
        img= Image.open('images/agent'+str(i+1)+'.png')
        img = img.resize((50, 50), Image.ANTIALIAS)
        img=ImageTk.PhotoImage(img)
        agentImages.append(img)

    blankLabel=Label(root, text="     ", borderwidth=6, padx=padding,pady=padding,relief="solid")

    # #Add Tasks
    taskList=[]

    for i in range(rows):
        row=[]
        row1=[]
        for j in range(cols):
            row1.append(0)
        gridLabels.append(row1)

## loop till you get a valid grid...
print("Initializing... ")
out, offlineTrainRes = ut.load_instance(rows, seed)
if out == None:
    sys.exit(1)

gridGraph = out['gridGraph']
adjList = out['adjList']
vertices = out['verts']
agentVertices = out['agnt_verts']
taskVertices = out['task_verts']
## truncate task list to accomodate lesser number of tasks
assert len(taskVertices) >= numTasks
if len(taskVertices) != numTasks:
    delete_inds = random.sample(range(len(taskVertices)), len(taskVertices)-numTasks)
    tasks = [taskVertices[i] for i in range(len(taskVertices)) if i not in delete_inds]
    taskVertices = tasks
assert len(taskVertices) == numTasks

print(gridGraph)
for i in range(len(taskVertices)):
    colors = colors + [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

N = len(vertices)
N = 2**(psi+1)

if visualizer:
    for i in range(rows):
        for j in range(cols):
            if gridGraph[i][j]==0:
                gridLabels[i][j]=Label(root, text="     ", borderwidth=6, bg='#333366', padx=padding,pady=padding,relief="solid")
                gridLabels[i][j].grid(row=i,column=j)

            else:
                gridLabels[i][j]=Label(root, text="     ", borderwidth=6, padx=padding,pady=padding,relief="solid")
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
        gridLabels[taskVertices[i][0]][taskVertices[i][1]]=Label(image=taskList[i],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[taskVertices[i][0]][taskVertices[i][1]].grid(row=taskVertices[i][0],column=taskVertices[i][1])

def changeCell(x,y,cellType,agentNum):
    sys.stdout.flush()
    gridLabels[x][y].grid_forget()
    sys.stdout.flush()
    
    if cellType=='task':
        gridLabels[x][y]=Label(image=taskList[0],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=r1,column=r2)
    elif cellType=='agent':
        gridLabels[x][y]=Label(image=agentImages[agentNum-1],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)
    elif cellType=='blank':
        gridLabels[x][y]=Label(root, text="     ", borderwidth=6,  padx=padding,pady=padding,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)

class Agent:
    def __init__(self,x,y,ID,color=1):
        self.posX=x
        self.posY=y
        self.prev_move = None
        self.cost=0
        if visualizer:
            changeCell(x,y,'agent',color)
        self.color=color
        self.ID=ID

        self.copy_number = 1

        self.posX_prime = x
        self.posY_prime = y
        self.cost_prime = 0
        self.color_prime = color

        self.gui_split = False

        self.exploring = False
        self.exp_dir = ''
        self.exp_dist_remaining = 0

        self.reset()

    def resetColor(self):
        self.color=11
        self.color_prime=11
        if visualizer:
            changeCell(self.posX,self.posY,'agent',self.color)
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

    def getCostIncurred(self):
        return self.cost

    def getDir(self):
        return self.dir

    def move(self,dir):
        self.dir=dir

    def updateView(self):
        self.viewEdges=set()
        self.viewEdges.add(((self.posX,self.posY),(self.posX,self.posY)))
        self.viewVertices=set([(self.posX,self.posY)])
        self.viewAgents=set()
        self.viewTasks=set()

        self.viewEdges_prime=set()
        self.viewEdges_prime.add(((self.posX,self.posY),(self.posX,self.posY)))
        self.viewVertices_prime=set([(self.posX,self.posY)])
        self.viewAgents_prime=set()
        self.viewTasks_prime=set()
        #Create Internal Representation
        for i in range(1,k+1):
            if (self.posX+i,self.posY) in vertices:
                #self.viewEdges.add(((self.posX,self.posY),(self.posX+i,self.posY)))
                #self.viewEdges.add(((self.posX+i,self.posY),(self.posX,self.posY)))
                self.viewVertices.add((self.posX+i,self.posY))
                self.viewVertices_prime.add((self.posX+i,self.posY))
                if (self.posX+i,self.posY) in taskVertices:
                    self.viewTasks.add((self.posX+i,self.posY))
                    self.viewTasks_prime.add((self.posX+i,self.posY))
                for j in  range(1,k-i+1):
                    if (self.posX+i,self.posY+j) in vertices:
                        #self.viewEdges.add(((self.posX+i,self.posY+j-1),(self.posX+i,self.posY+j)))
                        #self.viewEdges.add(((self.posX+i,self.posY+j),(self.posX+i,self.posY+j-1)))
                        self.viewVertices.add((self.posX+i,self.posY+j))
                        self.viewVertices_prime.add((self.posX+i,self.posY+j))
                        if (self.posX+i,self.posY+j) in taskVertices:
                            self.viewTasks.add((self.posX+i,self.posY+j))
                            self.viewTasks_prime.add((self.posX+i,self.posY+j))
                    if (self.posX+i,self.posY-j) in vertices:
                        #self.viewEdges.add(((self.posX+i,self.posY-j+1),(self.posX+i,self.posY-j)))
                        #self.viewEdges.add(((self.posX+i,self.posY-j),(self.posX+i,self.posY-j+1)))
                        self.viewVertices.add((self.posX+i,self.posY-j))
                        self.viewVertices_prime.add((self.posX+i,self.posY-j))
                        if (self.posX+i,self.posY-j) in taskVertices:
                            self.viewTasks.add((self.posX+i,self.posY-j))
                            self.viewTasks_prime.add((self.posX+i,self.posY-j))
            if (self.posX-i,self.posY) in vertices:
                #self.viewEdges.add(((self.posX,self.posY),(self.posX-i,self.posY)))
                #self.viewEdges.add(((self.posX-i,self.posY),(self.posX,self.posY)))
                self.viewVertices.add((self.posX-i,self.posY))
                self.viewVertices_prime.add((self.posX-i,self.posY))
                if (self.posX-i,self.posY) in taskVertices:
                    self.viewTasks.add((self.posX-i,self.posY))
                    self.viewTasks_prime.add((self.posX-i,self.posY))
                for j in  range(1,k-i+1):
                    if (self.posX-i,self.posY+j) in vertices:
                        #self.viewEdges.add(((self.posX-i,self.posY+j-1),(self.posX-i,self.posY+j)))
                        #self.viewEdges.add(((self.posX-i,self.posY+j),(self.posX-i,self.posY+j-1)))
                        self.viewVertices.add((self.posX-i,self.posY+j))
                        self.viewVertices_prime.add((self.posX-i,self.posY+j))
                        if (self.posX-i,self.posY+j) in taskVertices:
                            self.viewTasks.add((self.posX-i,self.posY+j))
                            self.viewTasks_prime.add((self.posX-i,self.posY+j))

                    if (self.posX-i,self.posY-j) in vertices:
                        #self.viewEdges.add(((self.posX-i,self.posY-j+1),(self.posX-i,self.posY-j)))
                        #self.viewEdges.add(((self.posX-i,self.posY-j),(self.posX-i,self.posY-j+1)))
                        self.viewVertices.add((self.posX-i,self.posY-j))
                        self.viewVertices_prime.add((self.posX-i,self.posY-j))
                        if (self.posX-i,self.posY-j) in taskVertices:
                            self.viewTasks.add((self.posX-i,self.posY-j))
                            self.viewTasks_prime.add((self.posX-i,self.posY-j))


            if (self.posX,self.posY+i) in vertices:
                #self.viewEdges.add(((self.posX,self.posY),(self.posX,self.posY+i)))
                # self.viewEdges.add(((self.posX,self.posY+i),(self.posX,self.posY)))
                self.viewVertices.add((self.posX,self.posY+i))
                self.viewVertices_prime.add((self.posX,self.posY+i))
                if (self.posX,self.posY+i) in taskVertices:
                    self.viewTasks.add((self.posX,self.posY+i))
                    self.viewTasks_prime.add((self.posX,self.posY+i))
                for j in  range(1,k-i+1):
                    if (self.posX+j,self.posY+i) in vertices:
                        #self.viewEdges.add(((self.posX+j-1,self.posY+i),(self.posX+j,self.posY+i)))
                        #self.viewEdges.add(((self.posX+j,self.posY+i),(self.posX+j-1,self.posY+i)))
                        self.viewVertices.add((self.posX+j,self.posY+i))
                        self.viewVertices_prime.add((self.posX+j,self.posY+i))
                        if (self.posX+j,self.posY+i) in taskVertices:
                            self.viewTasks.add((self.posX+j,self.posY+i))
                            self.viewTasks_prime.add((self.posX+j,self.posY+i))
                    if (self.posX-j,self.posY+i) in vertices:
                        #self.viewEdges.add(((self.posX-j+1,self.posY+i),(self.posX-j,self.posY+i)))
                        #self.viewEdges.add(((self.posX-j,self.posY+1),(self.posX-j+1,self.posY+i)))
                        self.viewVertices.add((self.posX-j,self.posY+i))
                        self.viewVertices_prime.add((self.posX-j,self.posY+i))
                        if (self.posX-j,self.posY+i) in taskVertices:
                            self.viewTasks.add((self.posX-j,self.posY+i))
                            self.viewTasks_prime.add((self.posX-j,self.posY+i))

            if (self.posX,self.posY-i) in vertices:
                #self.viewEdges.add(((self.posX,self.posY),(self.posX,self.posY-i)))
                #self.viewEdges.add(((self.posX,self.posY-i),(self.posX,self.posY)))
                self.viewVertices.add((self.posX,self.posY-i))
                self.viewVertices_prime.add((self.posX,self.posY-i))
                if (self.posX,self.posY-i) in taskVertices:
                    self.viewTasks.add((self.posX,self.posY-i))
                    self.viewTasks_prime.add((self.posX,self.posY-i))
                for j in  range(1,k-i+1):
                    if (self.posX+j,self.posY-i) in vertices:
                        # self.viewEdges.add(((self.posX+j-1,self.posY-i),(self.posX+j,self.posY-i)))
                        # self.viewEdges.add(((self.posX+j,self.posY-i),(self.posX+j-1,self.posY-i)))
                        self.viewVertices.add((self.posX+j,self.posY-i))
                        self.viewVertices_prime.add((self.posX+j,self.posY-i))
                        if (self.posX+j,self.posY-i) in taskVertices:
                            self.viewTasks.add((self.posX+j,self.posY-i))
                            self.viewTasks_prime.add((self.posX+j,self.posY-i))
                    if (self.posX-j,self.posY-i) in vertices:
                        #self.viewEdges.add(((self.posX-j+1,self.posY-i),(self.posX-j,self.posY-i)))
                        #self.viewEdges.add(((self.posX-j,self.posY+1),(self.posX-j+1,self.posY-i)))
                        self.viewVertices.add((self.posX-j,self.posY-i))
                        self.viewVertices_prime.add((self.posX-j,self.posY-i))
                        if (self.posX-j,self.posY-i) in taskVertices:
                            self.viewTasks.add((self.posX-j,self.posY-i))
                            self.viewTasks_prime.add((self.posX-j,self.posY-i))

            for u in self.viewVertices:
                if (u[0]+1,u[1]) in self.viewVertices:
                    self.viewEdges.add(((u[0],u[1]),(u[0]+1,u[1])))
                    self.viewEdges.add(((u[0]+1,u[1]),(u[0],u[1])))

                if (u[0]-1,u[1]) in self.viewVertices:
                    self.viewEdges.add(((u[0],u[1]),(u[0]-1,u[1])))
                    self.viewEdges.add(((u[0]-1,u[1]),(u[0],u[1])))

                if (u[0],u[1]+1) in self.viewVertices:
                    self.viewEdges.add(((u[0],u[1]),(u[0],u[1]+1)))
                    self.viewEdges.add(((u[0],u[1]+1),(u[0],u[1])))

                if (u[0],u[1]-1) in self.viewVertices:
                    self.viewEdges.add(((u[0],u[1]),(u[0],u[1]-1)))
                    self.viewEdges.add(((u[0],u[1]-1),(u[0],u[1])))

            for u in self.viewVertices_prime:
                if (u[0]+1,u[1]) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0],u[1]),(u[0]+1,u[1])))
                    self.viewEdges_prime.add(((u[0]+1,u[1]),(u[0],u[1])))

                if (u[0]-1,u[1]) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0],u[1]),(u[0]-1,u[1])))
                    self.viewEdges_prime.add(((u[0]-1,u[1]),(u[0],u[1])))

                if (u[0],u[1]+1) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0],u[1]),(u[0],u[1]+1)))
                    self.viewEdges_prime.add(((u[0],u[1]+1),(u[0],u[1])))

                if (u[0],u[1]-1) in self.viewVertices_prime:
                    self.viewEdges_prime.add(((u[0],u[1]),(u[0],u[1]-1)))
                    self.viewEdges_prime.add(((u[0],u[1]-1),(u[0],u[1])))

            s=self.viewVertices.copy()
            E=self.viewEdges.copy()
            T=self.viewTasks.copy()

            s_prime=self.viewVertices_prime.copy()
            E_prime=self.viewEdges_prime.copy()
            T_prime=self.viewTasks_prime.copy()
            ##print(E)
            for u in s:
                if u!=(self.posX,self.posY) and not bfs(s,E,(self.posX,self.posY),u):
                    for e in E:
                        if (e[0]==u or e[1]==u) and e in self.viewEdges:
                            self.viewEdges.remove(e)
                    self.viewVertices.remove(u)
                    if u in T:
                        self.viewTasks.remove(u)
            del s

            for u in s_prime:
                if u!=(self.posX,self.posY) and not bfs(s_prime,E_prime,(self.posX,self.posY),u):
                    for e in E_prime:
                        if (e[0]==u or e[1]==u) and e in self.viewEdges_prime:
                            self.viewEdges_prime.remove(e)
                    self.viewVertices_prime.remove(u)
                    if u in T_prime:
                        self.viewTasks_prime.remove(u)
            del s_prime

    def mapOffset(self,offX,offY,mapVerts,mapEdges,mapTasks,mapAgents):
        vertices=set()
        edges=set()
        taskSet=set()
        agentSet={}
        for v in mapVerts:
            vertices.add((v[0]-offX,v[1]-offY))
        for e in mapEdges:
            newEdge=((e[0][0]-offX,e[0][1]-offY),(e[1][0]-offX,e[1][1]-offY))
            edges.add(newEdge)
        for t in mapTasks:
            taskSet.add((t[0]-offX,t[1]-offY))
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

def updateAgentToAgentView():
    for a in agents:
        a.viewAgents.add(a)
        a.viewAgents_prime.add(a)
        for b in agents:
            if (b.posX,b.posY) in a.viewVertices:
                a.viewAgents.add(b)

            if (b.posX_prime,b.posY_prime) in a.viewVertices_prime:
                a.viewAgents_prime.add(b)

id = 1
agents = []
for i in range(A):
    agent = Agent(agentVertices[i][0], agentVertices[i][1], id, 11)
    agents.append(agent)
    print("{}: ({},{})".format(agent.ID, agent.posX, agent.posY))

    id += 1

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

def getExplorationMove(agent, lookupTable):
    legal = getLegalMovesFrom(agent.posX, agent.posY)
    if exp_strat == 0:
        if len(legal) > 1:
            legal.remove('q')
        return random.choice(legal)

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



def getLegalMovesFrom(x,y):
    moves=['q']
    if x+1 < rows and x+1 >= 0 and gridGraph[x+1][y] != 0:
        moves.append('e')
    if x-1 < rows and x-1 >= 0 and gridGraph[x-1][y] != 0:
        moves.append('w')
    if y+1 < cols and y+1 >= 0 and gridGraph[x][y+1] != 0:
        moves.append('n')
    if y-1 < cols and y-1 >= 0 and gridGraph[x][y-1] != 0:
        moves.append('s')
    return moves

