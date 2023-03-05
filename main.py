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
out, offlineTrainRes = load_instance(rows, seed)
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
    #time.sleep(1)
    gridLabels[x][y].grid_forget()
    sys.stdout.flush()
    # if x == 4 and y == 0:
    #   #print(cellType, agentNum)

    if cellType=='task':
        gridLabels[x][y]=Label(image=taskList[0],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=r1,column=r2)
    elif cellType=='agent':
        gridLabels[x][y]=Label(image=agentImages[agentNum-1],borderwidth=6, padx=6,pady=4.495,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)
    elif cellType=='blank':
        gridLabels[x][y]=Label(root, text="     ", borderwidth=6,  padx=padding,pady=padding,relief="solid")
        gridLabels[x][y].grid(row=x,column=y)



