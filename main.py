import sys, os
from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import pandas as pd
import random
import time
import itertools
import copy
import openpyxl
import threading

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

def stateUpdate():
    sys.stdout.flush()

    for a in agents:
        if visualizer:
            changeCell(a.posX, a.posY, 'blank', 0)

        if a.getDir() == 'e':
            a.setXPos(a.posX+1)
        elif a.getDir() == 'w':
            a.setXPos(a.posX-1)
        elif a.getDir() == 's':
            a.setYPos(a.posY-1)
        elif a.getDir() == 'n':
            a.setYPos(a.posY+1)

        if (a.posX, a.posY) in taskVertices:
            taskVertices.remove((a.posX, a.posY))

        sys.stdout.flush()

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

def multiAgentRollout(networkVertices,networkEdges,networkAgents,taskPos,agent,prevMoves):
    currentPos={}
    for a in networkAgents: ## a is actually a.ID...
        currentPos[a]=(agents[a-1].posX-networkAgents[a][0],agents[a-1].posY-networkAgents[a][1])
    currentTasks=taskPos.copy()

    prevCost=0
    for a_ID in networkAgents:
        # #print("moved", a_ID)
        a = agents[a_ID-1]
        if a in prevMoves:
            if prevMoves[a]=='n':
                currentPos[a_ID]=(currentPos[a_ID][0],currentPos[a_ID][1]+1)
            if prevMoves[a]=='s':
                currentPos[a_ID]=(currentPos[a_ID][0],currentPos[a_ID][1]-1)
            if prevMoves[a]=='e':
                currentPos[a_ID]=(currentPos[a_ID][0]+1,currentPos[a_ID][1])
            if prevMoves[a]=='w':
                currentPos[a_ID]=(currentPos[a_ID][0]-1,currentPos[a_ID][1])
            if prevMoves[a]=='q': ## wait move
                currentPos[a_ID]=(currentPos[a_ID][0], currentPos[a_ID][1])
            if currentPos[a_ID] in currentTasks:
                currentTasks.remove(currentPos[a_ID])

    agent_ID = list(agent.keys())[0]
    agent_posX = agents[agent_ID-1].posX - agent[agent_ID][0]
    agent_posY = agents[agent_ID-1].posY - agent[agent_ID][1]
    x_offset = agent[agent_ID][0]
    y_offset = agent[agent_ID][1]
    agent_prev_move = agents[agent_ID-1].prev_move
    if len(currentTasks)==0:
        return 'q', 0
    assert len(currentTasks)>0
    minCost=float('inf')
    bestMove=None
    Qfactors=[]
    #print(networkEdges)
    for e in networkEdges:
        assert e[1] in networkVertices
        if e[0]==(agent_posX,agent_posY):
            tempCurrentTasks=currentTasks.copy()
            tempPositions=currentPos.copy()
            if e[1] != e[0]:
                cost=prevCost+1
            else:
                cost=prevCost
            tempPositions[agent_ID]=e[1]
            if tempPositions[agent_ID] in tempCurrentTasks:
                # cost -= 100
                tempCurrentTasks.remove(tempPositions[agent_ID])

            rounds = 0
            while len(tempCurrentTasks)>0:
                if rounds >= len(networkVertices) + len(networkEdges):
                    cost += len(networkVertices) + len(networkEdges)
                    break
                for a in networkAgents:
                    if (a in prevMoves.keys() and prevMoves[a] != 'q') or (a not in prevMoves.keys()):
                        shortestDist=float('inf')
                        bestNewPos=None

                        for t in tempCurrentTasks:
                            a_pos = (tempPositions[a][0]+x_offset,tempPositions[a][1]+y_offset)
                            t_pos = (t[0]+x_offset,t[1]+y_offset)
                            dist, path = offlineTrainRes[str(a_pos)][str(t_pos)]
                            if dist<shortestDist:
                                shortestDist=dist
                                bestNewPos=path
                        if bestNewPos != None:
                            if (e[1] != (agent_posX,agent_posY)) or (agent_ID != a):
                                #print(a_pos, t_pos, bestNewPos, vertices)
                                tempPositions[a]=(bestNewPos[0]-x_offset,bestNewPos[1]-y_offset)
                                cost += 1
                        if tempPositions[a] in tempCurrentTasks:
                            # cost -= 100
                            tempCurrentTasks.remove(tempPositions[a])
                            #print("\tRemoving...", tempPositions[a], tempCurrentTasks)
                        rounds += 1
                        if len(tempCurrentTasks)==0:
                            break
            if cost<minCost:
                minCost=cost
                bestMove=e[1]

            Qfactors.append((e[1],cost))
            del tempPositions
            del tempCurrentTasks

    assert bestMove!=None
    minQ = float('inf')
    for factor in Qfactors:
        if factor[1] < minQ:
            minQ = factor[1]

    ## collect all ties...
    ties = []
    for factor in Qfactors:
         if factor[1] == minQ:
             if factor[0]==(agent_posX+1,agent_posY):
                 if (agent_prev_move == None) or (agent_prev_move != None and getOppositeDirection('e') != agent_prev_move):
                     ties.append((factor,'e'))
             elif factor[0]==(agent_posX-1,agent_posY):
                 if ((agent_prev_move == None) or (agent_prev_move != None and getOppositeDirection('w') != agent_prev_move)):
                     ties.append((factor,'w'))
             elif factor[0]==(agent_posX,agent_posY+1):
                 if (agent_prev_move == None) or (agent_prev_move != None and getOppositeDirection('s') != agent_prev_move):
                     ties.append((factor,'s'))
             elif factor[0]==(agent_posX,agent_posY-1):
                 if (agent_prev_move == None) or (agent_prev_move != None and getOppositeDirection('n') != agent_prev_move):
                     ties.append((factor,'n'))
             elif factor[0]==(agent_posX,agent_posY):
                 ties.append((factor,'q'))

    if len(ties) == 0:
         for factor in Qfactors:
             if factor[1] == minQ:
                 if factor[0]==(agent_posX+1,agent_posY):
                     ties.append((factor,'e'))
                 elif factor[0]==(agent_posX-1,agent_posY):
                     ties.append((factor,'w'))
                 elif factor[0]==(agent_posX,agent_posY+1):
                     ties.append((factor,'n'))
                 elif factor[0]==(agent_posX,agent_posY-1):
                     ties.append((factor,'s'))
                 elif factor[0]==(agent_posX,agent_posY):
                     ties.append(factor,'q')

    bestMove = ties[0][0][0]

    if bestMove==(agent_posX+1,agent_posY):
        ret= 'e'
    elif bestMove==(agent_posX-1,agent_posY):
        ret= 'w'
    elif bestMove==(agent_posX,agent_posY+1):
        ret= 'n'
    elif bestMove==(agent_posX,agent_posY-1):
        ret= 's'
    elif bestMove==(agent_posX,agent_posY):
        ret= 'q' ## wait move
    return ret,minCost

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

        if agent.gui_split == True:
            #print(agent.ID, agent.color)
            if visualizer:
                changeCell(agent.posX, agent.posY, "agent", agent.color)
            agent.gui_split = False

    ## Merge Mutual Connections
    for agent in agents:
        ## if x is not the parent of agent, then remove x from agent.children
        for x in agents:
            if x != agent and x != agent.parent:
                if agent in x.children:
                    x.children.remove(agent)

def getClosestClusterTask(agent, taskVertices, lookupTable):
    min_dist = float('inf')
    best_move = None
    for task in taskVertices:
        try:
            dist, path = lookupTable[str((agent.posX, agent.posY))][str(task)]
        except (AssertionError, KeyError):
            continue
        if dist < min_dist:
            min_dist = dist
            best_move = path
    if best_move == (agent.posX+1, agent.posY):
        next_dir = 'e'
    elif best_move == (agent.posX-1, agent.posY):
        next_dir = 'w'
    elif best_move == (agent.posX, agent.posY-1):
        next_dir = 's'
    elif best_move == (agent.posX, agent.posY+1):
        next_dir = 'n'
    elif best_move == None:
        next_dir = 'q'
    else:
        raise ValueError

    return next_dir, min_dist

def main():
    totalCost = 0
    waitCost = 0
    df = pd.DataFrame({'Centralized':[], 'Seed #': [],
                        'Rows': [], 'Cols': [], 'Wall Prob': [],
                        '# of Agents': [], '# of Tasks': [],
                        'k': [], 'psi': [], 'Total Time (s)': [],
                        '# of Exploration': []})
    lookupTable = offlineTrainRes

    if centralized:
        begin = time.time()
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
            if visualizer:
                costLabel = Label(root, text='Total Cost: ' + str(totalCost))
                costLabel.grid(row=rows+1, column=cols-3,columnspan=4)
        end = time.time()
        totalTime = end-begin
        print("Done. ")
        new_data['# of Exploration Steps'] = str(0)
        new_data['Wait Cost'] = str(waitCost)
        if visualizer:
            quitButton.invoke()

#Driver
def start():
    #print('starting')
    t=threading.Thread(target=main)
    t.start()

# if __name__ == "__main__":
#   main()
#Add Interface
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
    main()

