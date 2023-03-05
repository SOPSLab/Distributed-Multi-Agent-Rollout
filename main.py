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
