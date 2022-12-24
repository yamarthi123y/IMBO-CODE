# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:38:59 2022

@author: Desktop
"""

import os
import numpy as np
import matplotlib.pyplot as plt

x = [u'IGWO-SVM', u'ANN-CFS', u'RNN-ABC', u'GA-CNN',u'LSTM',u'Conv-LSTM-SFSDT',u'HCRNNIDS',u'PROPOSED']
z = [0.802,0.936,0.883,0.937,0.94,0.868,0.9491,0.9587]
y=[0.838,0.915,0.934,0.936,0.929,0.91,0.9387,0.9471]
#a=[2495,2879,3149,2345,3300]
#b=[1000,1200,1345,1500,1650]

fig, ax = plt.subplots(figsize =(12, 9))    
width = 0.3 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="#5C0099",label="Kappa")
ax.barh(ind+width,z, width, color="#009900",label="MCC")
#ax.barh(ind+width*2,a, width, color="#CC0066",label="as")
#ax.barh(ind+width*3,b, width, color="#009900",label="as")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False,fontsize=12,fontname="Times New Roman",fontweight='bold')
for i, v in enumerate(y):
    ax.text(v + 0.001, i + .01, str(v), color='black', fontweight='bold',fontsize=10)
for i, v in enumerate(z):
    ax.text(v + 0.001, i + .2, str(v), color='black', fontweight='bold',fontsize=10)
#for i, v in enumerate(a):
 #   ax.text(v + 3, i + .4, str(v), color='black', fontweight='bold',fontsize=7)
#for i, v in enumerate(b):
 #   ax.text(v + 3, i + .6, str(v), color='black', fontweight='bold',fontsize=7)            
#plt.title('title')
plt.xlim(0.5,1)
plt.xlabel('Methods',fontweight ='bold', fontsize = 18,fontname="Times New Roman")
#plt.ylabel('y')   
plt.legend(loc="lower left")
#---------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
x = [u'NSL-KD',u'CIC-IDS2017',u'CSE-CIC-IDS2018']
z = [0.937,0.936,0.949]
y=[0.921,0.938,0.9387]
#a=[2495,2879,3149,2345,3300]
#b=[1000,1200,1345,1500,1650]
fig, ax = plt.subplots(figsize =(8, 5))
width = 0.15 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="#800033",label="Kappa")
ax.barh(ind+width,z, width, color="#006600",label="MCC")
#ax.barh(ind+width*2,a, width, color="#CC0066",label="as")
#ax.barh(ind+width*3,b, width, color="#009900",label="as")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False,fontsize=12,fontname="Times New Roman",fontweight='bold')
for i, v in enumerate(y):
    ax.text(v + 0.001, i + (-.07), str(v), color='black', fontname="Times New Roman",fontsize=11)
for i, v in enumerate(z):
    ax.text(v + 0.001, i + .17, str(v), color='black', fontname="Times New Roman",fontsize=11)
#for i, v in enumerate(a):
 #   ax.text(v + 3, i + .4, str(v), color='black', fontweight='bold',fontsize=7)
#for i, v in enumerate(b):
 #   ax.text(v + 3, i + .6, str(v), color='black', fontweight='bold',fontsize=7)
#plt.title('title')
plt.xlim(0.5,1)
plt.xlabel('Methods',fontweight ='bold', fontsize = 18,fontname="Times New Roman")
#plt.ylabel('y')
plt.legend(loc="lower left")

#------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

x = [u'IGWO-SVM', u'ANN-CFS', u'RNN-ABC', u'GA-CNN',u'LSTM',u'Conv-LSTM-SFSDT',u'HCRNNIDS',u'ABC-BWO-CONV-LSTM',u'PROPOSED']
z = [0.323,0.699,0.831,0.787,0.843,0.8861,0.8798,0.927,0.934]
y=[0.804,0.644,0.82,0.831,0.885,0.896,0.9276,0.932,0.946]
#a=[2495,2879,3149,2345,3300]
#b=[1000,1200,1345,1500,1650]

fig, ax = plt.subplots(figsize =(12, 9))    
width = 0.3 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="#99CC00",label="Kappa")
ax.barh(ind+width,z, width, color="#FF0066",label="MCC")
#ax.barh(ind+width*2,a, width, color="#CC0066",label="as")
#ax.barh(ind+width*3,b, width, color="#009900",label="as")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False,fontsize=12,fontname="Times New Roman",fontweight='bold')
for i, v in enumerate(y):
    ax.text(v + 0.001, i + .01, str(v), color='black', fontweight='bold',fontsize=10)
for i, v in enumerate(z):
    ax.text(v + 0.001, i + .2, str(v), color='black', fontweight='bold',fontsize=10)
#for i, v in enumerate(a):
 #   ax.text(v + 3, i + .4, str(v), color='black', fontweight='bold',fontsize=7)
#for i, v in enumerate(b):
 #   ax.text(v + 3, i + .6, str(v), color='black', fontweight='bold',fontsize=7)            
#plt.title('title')
plt.xlim(0.2,1)
plt.xlabel('Methods',fontweight ='bold', fontsize = 18,fontname="Times New Roman")
#plt.ylabel('y')   
plt.legend(loc="lower right")
#----------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
x = [u'IGWO-SVM',u'ANN-CFS',u'RNN-ABC',u'GA-CNN',u'LSTM',u'Conv-LSTM-SFSDT',u'HCRNNIDS',u'ABC-BWO-CONV-LSTM',u'PROPOSED']
z = [0.719,0.765,0.841,0.897,0.883,0.8758,0.897,0.947,0.954]
y=[0.849,0.804,0.87,0.874,0.907,0.901,0.945,0.927,0.966]
#a=[2495,2879,3149,2345,3300]
#b=[1000,1200,1345,1500,1650]
fig, ax = plt.subplots(figsize =(10, 7))
width = 0.25 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="#4D9900",label="Kappa")
ax.barh(ind+width,z, width, color="#003366",label="MCC")
#ax.barh(ind+width*2,a, width, color="#CC0066",label="as")
#ax.barh(ind+width*3,b, width, color="#009900",label="as")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False,fontsize=12,fontname="Times New Roman",fontweight='bold')
for i, v in enumerate(y):
    ax.text(v + 0.001, i + (-.07), str(v), color='black', fontname="Times New Roman",fontsize=11)
for i, v in enumerate(z):
    ax.text(v + 0.001, i + .17, str(v), color='black', fontname="Times New Roman",fontsize=11)
#for i, v in enumerate(a):
 #   ax.text(v + 3, i + .4, str(v), color='black', fontweight='bold',fontsize=7)
#for i, v in enumerate(b):
 #   ax.text(v + 3, i + .6, str(v), color='black', fontweight='bold',fontsize=7)
#plt.title('title')
plt.xlim(0,1)
plt.xlabel('Methods',fontweight ='bold', fontsize = 18,fontname="Times New Roman")
#plt.ylabel('y')
plt.legend(loc="lower left")

