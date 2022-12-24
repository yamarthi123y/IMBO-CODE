# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:55:53 2022

@author: Desktop
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys



Paid = [91.45,96.6,96.65,97.49,96.21,97.27,98.25,98.89] 
Unpaid = [80.12,82.71,89.03,91.78,88.38,94.91,97.48,98.12]
poi=[90.81,93.75,97.74,96.84,97.14,97.79,98.67,98.82]
timo=[93.19,97.06,99.08,96.41,97.4,97.09,98.18,98.75]
Paid_Percentages = [91.45,96.6,96.65,97.49,96.21,97.27,98.25,98.89]
Unpaid_Percentages = [80.12,82.71,89.03,91.78,88.38,94.91,97.48,98.12]
poi_Percentages=[90.81,93.75,97.74,96.84,97.14,97.79,98.67,98.82]
timo_Percentages=[93.19,97.06,99.08,96.41,97.4,97.09,98.18,98.75]
n=8
r = np.arange(n)
width = 0.15
#ax,fig = plt.figure(figsize = (12, 8))
fig, ax = plt.subplots(figsize = (8, 6))
rects1 = ax.bar(r, Paid, color = '#5C0099', width = width, edgecolor = 'black', label='Accuracy')
rects2 = ax.bar(r + width, Unpaid, color = '#E69900', width = width, edgecolor = 'black', label='Precision')
rects3 = ax.bar(r + width*2, poi, color = '#002B80', width = width, edgecolor = 'black', label='Recall')
rects4 = ax.bar(r + width*3, timo, color = '#99003D', width = width, edgecolor = 'black', label='F-Measure')

#plt.grid(color='gray',ls='dotted',linewidth=1)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
#plt.xlabel("Methods")
plt.ylabel('Percentage(%)',fontweight ='bold')
#plt.title("Comparative Performance of Feature Selection Algorithm",fontweight ='bold', fontsize = 17,fontname="Times New Roman")
plt.ylim(75,105)
plt.yticks(np.arange(75, 105, step=5))
plt.xticks(r + width,['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','PROPOSED'],fontsize=12,rotation=50)
plt.legend(loc ="lower right")

for rect,p in zip(rects1, Paid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')

for rect,p in zip(rects2, Unpaid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
for rect,p in zip(rects3,poi_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
for rect,p in zip(rects4,timo_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
Paid = [98.97,98.13,98.73,97.48]
Unpaid = [99.03,95.77,98.65,98.54]
poi=[98.87,97.48,98.18,97.84]
#timo=[95.60,96.18,96.18,97.32,99.38,99.89]
#doli=[99.67,	98.39,	99.05]
Paid_Percentages = [98.97,98.13,98.73,97.48]
Unpaid_Percentages = [99.03,95.77,98.65,98.54]
poi_Percentages=[98.87,97.48,98.18,97.84]
#timo_Percentages=[95.60,96.18,96.18,97.32,99.38,99.89]
#doli_Percentages=[99.67,	98.39,	99.05]

n=4
r = np.arange(n)
width = 0.18
#ax,fig = plt.figure(figsize = (12, 8))
fig, ax = plt.subplots(figsize = (8, 5))
rects1 = ax.bar(r, Paid, color = '#00b33c', width = width, edgecolor = 'black', label='NSL-KDD')
rects2 = ax.bar(r + width, Unpaid, color = '#668cff', width = width, edgecolor = 'black', label='CIC-IDS2017')
rects3 = ax.bar(r + width*2, poi, color = '#ff6666', width = width, edgecolor = 'black', label='CSE-CIC-IDS2018')
#rects4 = ax.bar(r + width*3, timo, color = '#00B377', width = width, edgecolor = 'black', label='F1-Score')
#rects5 = ax.bar(r + width*4, doli, color = '#990099', width = width, edgecolor = 'black', label='EfficientNet')
#plt.grid(color='gray',ls='dotted',linewidth=1)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
plt.xlabel("Models",fontweight ='bold',fontsize=15,fontname="Times New Roman")
plt.ylabel('Performance (%)',fontweight ='bold',fontsize=15,fontname="Times New Roman")
#plt.title("Ternary classification",fontweight ='bold', fontsize = 17,fontname="Times New Roman")
plt.ylim(80,110)
plt.yticks(np.arange(80, 110, step=10))
plt.xticks(r + width,['Accuracy','Precision','Recall ','F-Measure'],fontsize=12,rotation=35)
plt.legend(loc ="upper left")
for rect,p in zip(rects1, Paid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
for rect,p in zip(rects2, Unpaid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
for rect,p in zip(rects3,poi_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
#for rect,p in zip(rects4,timo_Percentages):
    #height = rect.get_height()
    #ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
#for rect,p in zip(rects5,doli_Percentages):
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width()/1, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=55, color='black',fontweight='bold')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys



Paid = [85.76,81.66,90.29,93.89,89.94,92.11,95.35,98.67,99.02] 
Unpaid = [90.62,69.89,80.86,86.45,88.88,90.02,96.67,97.48,98.67]
poi=[91.09,85.43,96.01,92.37,94.57,93.44,97.48,96.67,98.82]
timo=[85.36,74.32,90.61,91.02,95.75,94.56,96.5,98.73,98.73]
Paid_Percentages = [85.76,81.66,90.29,93.89,89.94,92.11,95.35,98.67,99.02]
Unpaid_Percentages = [90.62,69.89,80.86,86.45,88.88,90.02,96.67,97.48,98.67]
poi_Percentages=[91.09,85.43,96.01,92.37,94.57,93.44,97.48,96.67,98.82]
timo_Percentages=[85.36,74.32,90.61,91.02,95.75,94.56,96.5,98.73,98.73]
n=9
r = np.arange(n)
width = 0.18
#ax,fig = plt.figure(figsize = (12, 8))
fig, ax = plt.subplots(figsize = (8, 6))
rects1 = ax.bar(r, Paid, color = '#999900', width = width, edgecolor = 'black', label='Accuracy')
rects2 = ax.bar(r + width, Unpaid, color = '#004D4D', width = width, edgecolor = 'black', label='Precision')
rects3 = ax.bar(r + width*2, poi, color = '#5C5C3D', width = width, edgecolor = 'black', label='Recall')
rects4 = ax.bar(r + width*3, timo, color = '#660066', width = width, edgecolor = 'black', label='F-Measure')

#plt.grid(color='gray',ls='dotted',linewidth=1)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
#plt.xlabel("Methods")
plt.ylabel('Percentage(%)',fontweight ='bold')
#plt.title("Comparative Performance of Feature Selection Algorithm",fontweight ='bold', fontsize = 17,fontname="Times New Roman")
plt.ylim(60,110)
plt.yticks(np.arange(60, 110, step=10))
plt.xticks(r + width*(-1),['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','ABC-BWO-CONV-LSTM','PROPOSED'],fontsize=12,rotation=50)
plt.legend(loc ="lower right")

for rect,p in zip(rects1, Paid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')

for rect,p in zip(rects2, Unpaid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
for rect,p in zip(rects3,poi_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
for rect,p in zip(rects4,timo_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=90, color='black',fontweight='bold')
plt.show()

#-------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
Paid = [86.74,93.17,96.41,94.72,91.56,95.63,90.87,98.67,99.09]
Unpaid = [91.29,87.97,87.56,98.36,97.45,88.47,93.06,95.35,98.97]
poi=[94.56,92.17,94.56,91.02,93.44,92.37,96.5,97.48,98.74]
timo=[97.48,90.61,96.75,93.86,94.56,90.87,97.54,98.74,99.13]
#doli=[99.67,	98.39,	99.05]
Paid_Percentages = [86.74,93.17,96.41,94.72,91.56,95.63,90.87,98.67,99.09]
Unpaid_Percentages = [91.29,87.97,87.56,98.36,97.45,88.47,93.06,95.35,98.97]
poi_Percentages=[94.56,92.17,94.56,91.02,93.44,92.37,96.5,97.48,98.74]
timo_Percentages=[97.48,90.61,96.75,93.86,94.56,90.87,97.54,98.74,99.13]
#doli_Percentages=[99.67,	98.39,	99.05]

n=9
r = np.arange(n)
width = 0.18
#ax,fig = plt.figure(figsize = (12, 8))
fig, ax = plt.subplots(figsize = (8, 5))
rects1 = ax.bar(r, Paid, color = '#4D9900', width = width, edgecolor = 'black', label='Accuracy')
rects2 = ax.bar(r + width, Unpaid, color = '#003366', width = width, edgecolor = 'black', label='Precision')
rects3 = ax.bar(r + width*2, poi, color = '#CC0066', width = width, edgecolor = 'black', label='Recall')
rects4 = ax.bar(r + width*3, timo, color = '#B37700', width = width, edgecolor = 'black', label='F-Measure')
#rects5 = ax.bar(r + width*4, doli, color = '#990099', width = width, edgecolor = 'black', label='EfficientNet')
#plt.grid(color='gray',ls='dotted',linewidth=1)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
plt.xlabel("Models",fontweight ='bold',fontsize=15,fontname="Times New Roman")
plt.ylabel('Performance (%)',fontweight ='bold',fontsize=15,fontname="Times New Roman")
#plt.title("Ternary classification",fontweight ='bold', fontsize = 17,fontname="Times New Roman")
plt.ylim(80,110)
plt.yticks(np.arange(80, 110, step=10))
plt.xticks(r + width,['IGWO-SVM','ANN-CFS','RNN-ABC','GA-CNN','LSTM','Conv-LSTM-SFSDT','HCRNNIDS','ABC-BWO-CONV-LSTM','PROPOSED'],fontsize=12,rotation=35)
plt.legend(loc ="upper left")
for rect,p in zip(rects1, Paid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
for rect,p in zip(rects2, Unpaid_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
for rect,p in zip(rects3,poi_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
for rect,p in zip(rects4,timo_Percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/1.5, height+1, str(p), ha='center', va='bottom', fontsize=11, rotation=90, color='black',fontname="Times New Roman")
#for rect,p in zip(rects5,doli_Percentages):
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width()/1, height+1, str(p), ha='center', va='bottom', fontsize=9, rotation=55, color='black',fontweight='bold')
plt.show()
