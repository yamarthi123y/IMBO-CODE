# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:27:46 2022

@author: Desktop
"""

import matplotlib.pyplot as plt
import numpy as np
import re
fig,ax = plt.subplots(figsize =(8,6))
xlabels = ['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','PROPOSED']

#xlabels_new = [label.replace(' ', '\n') for label in xlabels]
A= [31.64,21.22,17.27,13.2,9.09,5.58,4.52,4.15]
B=[28.79,13.92,10.71,5.52,4.87,4.49,3.87,2.67]
plt.plot(xlabels, A, color='black', markerfacecolor='#CC0066', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FPR')
plt.plot(xlabels, B, color='black', markerfacecolor='#009900', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FNR')

ax.plot(xlabels,A)
label_x = xlabels[-8:]
label_y = A[-8:]
for i, txt in enumerate(label_y):
    ax.annotate(txt, (label_x[i], label_y[i]+2),fontweight="bold")
ax.plot(xlabels,B)
label_x = xlabels[-8:]
label_y = B[-8:]
for i, txt in enumerate(label_y):
        ax.annotate(txt, (label_x[i], label_y[i]+(-2)),fontweight="bold")
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)

			
plt.ylim(0,35)
plt.yticks(np.linspace(0,35,8))
plt.xlabel('Methods', fontsize=22,fontweight ='bold',fontname="Times New Roman",labelpad=14)
#plt.ylabel(('Performance Metrices(%)'), fontsize=22,fontweight ='bold',fontname="Times New Roman")
#plt.plot(range(5))
plt.xticks(range(8),fontname="Times New Roman",fontsize=12,rotation=50,fontweight="bold")
#plt.grid(axis='y',color = 'black', linestyle = '-', linewidth = 0.3)
plt.legend(loc='upper right')
plt.show()

#----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import re
fig,ax = plt.subplots(figsize =(8,6))
xlabels = ['NSL-KDD','CIC-IDS2017','CSE-CIC-IDS2018']
#xlabels_new = [label.replace(' ', '\n') for label in xlabels]
A= [7.5,4.68,5.74]
B=[5.02,3.8,1.67]
plt.plot(xlabels, A, color='black', markerfacecolor='#CC0066', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FPR')
plt.plot(xlabels, B, color='black', markerfacecolor='#009900', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FNR')
ax.plot(xlabels,A)
label_x = xlabels[-3:]
label_y = A[-3:]
for i, txt in enumerate(label_y):
    ax.annotate(txt, (label_x[i], label_y[i]+1),fontweight="bold")
ax.plot(xlabels,B)
label_x = xlabels[-3:]
label_y = B[-3:]
for i, txt in enumerate(label_y):
        ax.annotate(txt, (label_x[i], label_y[i]+(-1)),fontweight="bold")
ax.spines['right'].set_linewidth(-1)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
			
plt.ylim(0,10)
plt.yticks(np.linspace(0,10,11))

plt.xlabel('Methods', fontsize=22,fontweight ='bold',fontname="Times New Roman",labelpad=14)
#plt.ylabel(('Performance Metrices(%)'), fontsize=22,fontweight ='bold',fontname="Times New Roman")
#plt.plot(range(5))
plt.xticks(range(3),fontname="Times New Roman",fontsize=12,rotation=20,fontweight="bold")
#plt.grid(axis='y',color = 'black', linestyle = '-', linewidth = 0.3)
plt.legend(loc='upper right')
plt.show()

#--------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import re
fig,ax = plt.subplots(figsize =(8,6))
xlabels = ['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','ABC-BWO-CONV-LSTM','PROPOSED']

#xlabels_new = [label.replace(' ', '\n') for label in xlabels]
A= [19.25,15.27,10.11,9.75,9.27,8.81,9.76,7.5,6.2]
B=[27.4,13.26,5.46,6.47,4.74,5.56,6.98,5.01,4.86]
plt.plot(xlabels, A, color='black', markerfacecolor='#B37700', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FPR')
plt.plot(xlabels, B, color='black', markerfacecolor='#007ACC', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FNR')

ax.plot(xlabels,A)
label_x = xlabels[-9:]
label_y = A[-9:]
for i, txt in enumerate(label_y):
    ax.annotate(txt, (label_x[i], label_y[i]+2),fontweight="bold")
ax.plot(xlabels,B)
label_x = xlabels[-9:]
label_y = B[-9:]
for i, txt in enumerate(label_y):
        ax.annotate(txt, (label_x[i], label_y[i]+(-2)),fontweight="bold")
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)

			
plt.ylim(0,35)
plt.yticks(np.linspace(0,35,8))
plt.xlabel('Methods', fontsize=22,fontweight ='bold',fontname="Times New Roman",labelpad=14)
#plt.ylabel(('Performance Metrices(%)'), fontsize=22,fontweight ='bold',fontname="Times New Roman")
#plt.plot(range(5))
plt.xticks(range(9),fontname="Times New Roman",fontsize=12,rotation=50,fontweight="bold")
#plt.grid(axis='y',color = 'black', linestyle = '-', linewidth = 0.3)
plt.legend(loc='upper right')
plt.show()

#-----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import re
fig,ax = plt.subplots(figsize =(8,6))
xlabels = ['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','ABC-BWO-CONV-LSTM','PROPOSED']
#xlabels_new = [label.replace(' ', '\n') for label in xlabels]
A= [21.25,18.29,10.47,12.7,9.21,10.81,9.76,6.5,5.2]
B=[25.34,14.85,5.46,6.47,7.89,4.87,7.68,5.01,4.12]
plt.plot(xlabels, A, color='#800033', markerfacecolor='#CC0066', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FPR')
plt.plot(xlabels, B, color='#006600', markerfacecolor='#009900', markersize=12, marker='o',linewidth=2.5,linestyle='dashed',label='FNR')
ax.plot(xlabels,A)
label_x = xlabels[-9:]
label_y = A[-9:]
for i, txt in enumerate(label_y):
    ax.annotate(txt, (label_x[i], label_y[i]+1),fontweight="bold")
ax.plot(xlabels,B)
label_x = xlabels[-9:]
label_y = B[-9:]
for i, txt in enumerate(label_y):
        ax.annotate(txt, (label_x[i], label_y[i]+(-1)),fontweight="bold")
ax.spines['right'].set_linewidth(-1)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)  # setting up Y-axis tick color to red
ax.spines['bottom'].set_linewidth(2)
			
plt.ylim(0,30)
plt.yticks(np.linspace(0,30,11))

plt.xlabel('Methods', fontsize=22,fontweight ='bold',fontname="Times New Roman",labelpad=14)
#plt.ylabel(('Performance Metrices(%)'), fontsize=22,fontweight ='bold',fontname="Times New Roman")
#plt.plot(range(5))
plt.xticks(range(9),fontname="Times New Roman",fontsize=12,rotation=20,fontweight="bold")
#plt.grid(axis='y',color = 'black', linestyle = '-', linewidth = 0.3)
plt.legend(loc='upper right')
plt.show()
