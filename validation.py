# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:53:21 2022

@author: Kumar
"""

import pandas as pd
#import trntest
#import val
#import val
# import conf

matrix ={'tn': 0, 'tp': 10000, 'fn': 0, 'fp': 100}
'''data = pd.read_csv("__pycache__/files/param.dat")
print(data)
def predict(df,y,thresh_hold):
    y_pred=[]
    for label in df[y]:
        if label<thresh_hold:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred'''    

data = pd.read_csv("__pycache__/files/data2.dat")
print(data)
def Accuracy(tp_count, fp_count):
   
    return a_ab(tp_count, fp_count)

def Recall(tp_count, fn_count):
    
    return a_ab(tp_count, fn_count)


def Precision(tn_count, fp_count):
   
    
    return a_ab(tn_count, fp_count)


def F1_Score(tp_count, fp_count, fn_count, tn_count):
   
    
    summ = (tp_count + tn_count + fp_count + fn_count)
    return (float(tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)) if summ > 0 else 0


def cal_vals(df,y,y_pred):
    tp=()
    tn=()
    fn=()
    fp=()
    
    for val1,val2 in enumerate(df['y']):
        if(df.y_pred[val1]==1) and df.y[val1]==1:
            tp=tp+1
        if(df.y_pred[val1]==0) and df.y[val1]==0:
            tn=tn+1
        if(df.y_pred[val1]==0) and df.y[val1]==1:
            fn=fn+1
        if(df.y_pred[val1]==1) and df.y[val1]==0:
            fp=fp+1
        
    return {'tn':tn,'tp':tp,'fn':fn,'fp':fp}

Accuracy=(matrix['tp']+matrix['tn'])
Recall=matrix['tp']+(matrix['tp']+matrix['fp'])
Precision=matrix['tp']
F1_Score=(matrix["tp"])

#AUC=matrix['tn']
