# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:23 2022

@author: Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

Monday_path = 'dataset/CICIDS2017/pro-Monday-0.5v2.csv'
Monday = pd.read_csv(Monday_path ,sep=",")

Tuesday_path = 'dataset/CICIDS2017/pro-Tuesday-0.5v2.csv'
Tuesday = pd.read_csv(Tuesday_path ,sep=",")

Wednesday_path = 'dataset/CICIDS2017/pro-Wednesday-0.5v2.csv'
Wednesday = pd.read_csv(Wednesday_path ,sep=",")

Thursday_path = 'dataset/CICIDS2017/pro-Thursday-0.5v2.csv'
Thursday = pd.read_csv(Thursday_path ,sep=",")

Friday_path = 'dataset/CICIDS2017/pro-Friday-0.5v2.csv'
Friday = pd.read_csv(Friday_path ,sep=",")
def drop_attacks(data):

    drop_idx_master = []
    drop_idx = []
    for i in range(len(data)):
        if data.iloc[i]["Label"] == 'BENIGN':
            counter = 0
        elif data.iloc[i]["Label"] != 'BENIGN':
            counter += 1
            drop_idx.append(i)
            if data.iloc[i+1]["Label"] == 'BENIGN':
                amt_drop = int(0.95 * counter) # drop index of last 95%  of attack
                print ("{} initial amount: {}".format(data.iloc[i]["Label"], counter))
                print ("amt_dropped: " , amt_drop)
                drop_idx = drop_idx[-amt_drop:] # indexes list of one paticular attack to drop 
                drop_idx_master += drop_idx
                drop_idx = []

    data_reduced = data.drop(drop_idx_master)
    print ("drop completed!!")
    return data_reduced
def clean_data(data):
    
    data_label = data[["Label"]].copy()
    
    data_trimed = data.iloc[:, : len(data.columns) -1 ].copy() #exclude the 'Label' columns
    
    data_trimed = data_trimed.apply(pd.to_numeric,errors='coerce')
    data_trimed = data_trimed.fillna(data_trimed.mean())
    data_trimed = data_trimed.fillna(0.0)
    data_label.loc[data_label['Label'] == 'BENIGN' , "num_label"] = 0.0 #create new col "num_label" for numeric label
    data_label.loc[data_label['Label'] != 'BENIGN' , "num_label"] = 1.0
    
    data_final = pd.concat([data_label['num_label'], data_trimed], axis = 1)
    data_final = data_final.round(5)
    
    return data_final
def feature_select(meta_train, meta_valid, correlation_mark = 0.2):

    meta = pd.concat([meta_train, meta_valid], axis = 0)
    meta = clean_data(meta)

    meta_corr = meta.corr(method='pearson').iloc[:, [0]]
    meta_corr = meta_corr.fillna(0)
    features  = list(meta_corr.index)[:] 


    


    ############## feature selection #################
    feature_select_dict = dict()
    features_select = []
    for i in range(len(list(meta_corr.iloc[:,0]))):
        if abs(list(meta_corr.iloc[:,0])[i]) >= correlation_mark:
            feature_select_dict[features[i]] = list(meta_corr.iloc[:,0])[i]
            features_select.append(features[i])

    print ('number of features: {}'.format(len(features_select)))
    print ('\n')
    print (' Selected features for model: ', features_select)
    meta_train = clean_data(meta_train)
    meta_valid = clean_data(meta_valid)

    meta_train_select = meta_train[features_select]
    meta_valid_select = meta_valid[features_select]

    return meta_train_select, meta_valid_select, features_select, feature_select_dict
Tuesday_reduced = drop_attacks(Tuesday)
Wednesday_reduced = drop_attacks(Wednesday)
Thursday_reduced = drop_attacks(Thursday)
Friday_reduced = drop_attacks(Friday)
all_train_reduced = pd.concat([Monday, Tuesday_reduced, Wednesday_reduced], axis = 0)
all_valid_reduced = pd.concat([Thursday_reduced], axis = 0)
all_test_reduced = pd.concat([Friday_reduced], axis = 0)
print (all_train_reduced.groupby(['Label']).size().reset_index(name='counts'))
print (all_valid_reduced.groupby(['Label']).size().reset_index(name='counts'))
print (all_test_reduced.groupby(['Label']).size().reset_index(name='counts'))
all_attack = 360+360+78+102+126+138+120+42+24+246+240+12+120+360+120+228+240.0
non_attack = 151114 + 44133 + 38960

print (all_attack / (all_attack + non_attack))
print (all_attack + non_attack)
##****** Features used for the 1% attack datasets

threshold = 0.078 #select features with PCC score greater or equals to threshold 
alltrainreduced_select, allvalidreduced_select, features_select, fea_dict = feature_select(all_train_reduced, all_valid_reduced, correlation_mark = threshold)
alltestreduced_select = clean_data(all_test_reduced)[features_select]

fea_dict_d = sorted(fea_dict.items(), key=operator.itemgetter(1))
fea_dict_d  #Features and its PCC score sorted

#-------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import metrices
from sklearn.ensemble import RandomForestClassifier
import val
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
import line
# check the available data
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        #!nvidia-smi
     
#load the data into memory
network_data = pd.read_csv('dataset/IDS2018/02-14-2018.csv')
# check the shape of data
network_data.shape
# check the number of rows and columns
print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
print('Number of Columns (Features): %s' % str((network_data.shape[1])))
network_data.head(4)
# check the columns in data
network_data.columns
# check the number of columns
print('Total columns in our data: %s' % str(len(network_data.columns)))
network_data.info()
# check the number of values for labels
network_data['Label'].value_counts()
# make a plot number of labels
#sns.set(rc={'figure.figsize':(12, 6)})
#plt.xlabel('Attack Type')

#ax = sns.countplot(x='Label', data=network_data)
#ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
#plt.show()
# make a scatterplot
pyo.init_notebook_mode()
#fig = px.scatter(x = network_data["Bwd Pkts/s"][:100000], 
                # y=network_data["Fwd Seg Size Min"][:100000])
#fig

#sns.set(rc={'figure.figsize':(12, 6)})
#sns.scatterplot(x=network_data['Bwd Pkts/s'][:50000], y=network_data['Fwd Seg Size Min'][:50000], 
               # hue='Label', data=network_data)
# check the dtype of timestamp column
(network_data['Timestamp'].dtype)
# check for some null or missing values in our dataset
network_data.isna().sum().to_numpy()
# drop null or missing columns
cleaned_data = network_data.dropna()
cleaned_data.isna().sum().to_numpy()
# encode the column labels
label_encoder = LabelEncoder()
cleaned_data['Label']= label_encoder.fit_transform(cleaned_data['Label'])
cleaned_data['Label'].unique()
# check for encoded labels
cleaned_data['Label'].value_counts()
# make 3 seperate datasets for 3 feature labels
data_1 = cleaned_data[cleaned_data['Label'] == 0]
data_2 = cleaned_data[cleaned_data['Label'] == 1]
data_3 = cleaned_data[cleaned_data['Label'] == 2]

# make benign feature
y_1 = np.zeros(data_1.shape[0])
y_benign = pd.DataFrame(y_1)

# make bruteforce feature
y_2 = np.ones(data_2.shape[0])
y_bf = pd.DataFrame(y_2)

# make bruteforceSSH feature
y_3 = np.full(data_3.shape[0], 2)
y_ssh = pd.DataFrame(y_3)

# merging the original dataframe
X = pd.concat([data_1, data_2, data_3], sort=True)
y = pd.concat([y_benign, y_bf, y_ssh], sort=True)
y_1, y_2, y_3
print(X.shape)
print(y.shape)
# checking if there are some null values in data
X.isnull().sum().to_numpy()
from sklearn.utils import resample

data_1_resample = resample(data_1, n_samples=20000, 
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=20000, 
                           random_state=123, replace=True)
data_3_resample = resample(data_3, n_samples=20000, 
                           random_state=123, replace=True)
train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
train_dataset.head(2)
# viewing the distribution of intrusion attacks in our dataset 
#plt.figure(figsize=(10, 8))
#circle = plt.Circle((0, 0), 0.7, color='white')
#plt.title('Intrusion Attack Type Distribution')
#plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'BF', 'BF-SSH'], colors=['blue', 'magenta', 'cyan'])
#p = plt.gcf()
#p.gca().add_artist(circle)
test_dataset = train_dataset.sample(frac=0.1)
target_train = train_dataset['Label']
target_test = test_dataset['Label']
target_train.unique(), target_test.unique()
y_train = to_categorical(target_train, num_classes=3)
y_test = to_categorical(target_test, num_classes=3)
train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)
test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)
# making train & test splits
X_train = train_dataset.iloc[:, :-1].values
X_test = test_dataset.iloc[:, :-1].values
X_test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# reshape the data for CNN
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train.shape, X_test.shape
# making the deep learning function
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = model()
model.summary()
logger = CSVLogger('logs.csv', append=True)
his = model.fit(X_train, y_train, epochs=80, batch_size=32, 
          validation_data=(X_test, y_test), callbacks=[logger])
# check the model performance on test data
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save("model.h5")
print("Saved model to disk")
# check history of model
history = his.history
history.keys()
epochs = range(1, len(history['loss']) + 1)
acc = history['accuracy']
loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

# visualize training and val accuracy


#------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import metrics
import metrice
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
import roc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
import tnerset
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
import squezenet
#from mlxtend.plotting import plot_confusion_matrix

# Turn off the warnings.
warnings.filterwarnings(action='ignore')

Trained_Data = pd.read_csv("dataset/NSL-KDD/KDDTrain+.txt" , sep = "," , encoding = 'utf-8')
Tested_Data  = pd.read_csv("dataset/NSL-KDD/KDDTest+.txt" , sep = "," , encoding = 'utf-8')
Trained_Data
Tested_Data
Columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
            'dst_host_srv_rerro_rate','attack','level'])
Trained_Data.columns = Columns
Tested_Data.columns  = Columns
Trained_Data.head(10)
Tested_Data.head(10)
Trained_Data.info()
Tested_Data.info()
Trained_Data.describe()
Tested_Data.describe()
Trained_Data.nunique()
Tested_Data.nunique()
Trained_Data.max()
Tested_Data.max()
Results = set(Trained_Data['attack'].values)
print(Results,end=" ")
Trained_attack = Trained_Data.attack.map(lambda a: 0 if a == 'normal' else 1)
Tested_attack = Tested_Data.attack.map(lambda a: 0 if a == 'normal' else 1)
Trained_Data['attack_state'] = Trained_attack
Tested_Data['attack_state'] = Tested_attack
Trained_Data.head(10)
Tested_Data.head(10)
Trained_Data.isnull().sum()
Tested_Data.isnull().sum()
Trained_Data.duplicated().sum()
Tested_Data.duplicated().sum()
Trained_Data.shape

Trained_Data = pd.get_dummies(Trained_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
Tested_Data = pd.get_dummies(Tested_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
LE = LabelEncoder()
attack_LE= LabelEncoder()
Trained_Data['attack'] = attack_LE.fit_transform(Trained_Data["attack"])
Tested_Data['attack'] = attack_LE.fit_transform(Tested_Data["attack"])
X_train = Trained_Data.drop('attack', axis = 1)
X_train = Trained_Data.drop('level', axis = 1)
X_train = Trained_Data.drop('attack_state', axis = 1)
X_test = Tested_Data.drop('attack', axis = 1)
X_test = Tested_Data.drop('level', axis = 1)
X_test = Tested_Data.drop('attack_state', axis = 1)
Y_train = Trained_Data['attack_state']
Y_test = Tested_Data['attack_state']
X_train_train,X_test_train ,Y_train_train,Y_test_train = train_test_split(X_train, Y_train, test_size= 0.25 , random_state=42)
X_train_test,X_test_test,Y_train_test,Y_test_test = train_test_split(X_test, Y_test, test_size= 0.25 , random_state=42)
Ro_scaler = RobustScaler()
X_train_train = Ro_scaler.fit_transform(X_train_train) 
X_test_train= Ro_scaler.transform(X_test_train)
X_train_test = Ro_scaler.fit_transform(X_train_test) 
X_test_test= Ro_scaler.transform(X_test_test)
X_train_train.shape, Y_train_train.shape
X_test_train.shape, Y_test_train.shape
X_train_test.shape, Y_train_test.shape
X_test_test.shape, Y_test_test.shape
A = sm.add_constant(X_train)
Est1 = sm.GLM(Y_train, A)
Est2 = Est1.fit()
Est2.summary()
def Evaluate(Model_Name, Model_Abb, X_test, Y_test):
    
    Pred_Value= Model_Abb.predict(X_test)
    Accuracy = metrics.accuracy_score(Y_test,Pred_Value)                      
    Sensitivity = metrics.recall_score(Y_test,Pred_Value)
    Precision = metrics.precision_score(Y_test,Pred_Value)
    F1_score = metrics.f1_score(Y_test,Pred_Value)
    Recall = metrics.recall_score(Y_test,Pred_Value)
    
    print('--------------------------------------------------\n')
    print('The {} Model Accuracy   = {}\n'.format(Model_Name, np.round(Accuracy,3)))
    print('The {} Model Sensitvity = {}\n'.format(Model_Name, np.round(Sensitivity,3)))
    print('The {} Model Precision  = {}\n'.format(Model_Name, np.round(Precision,3)))
    print('The {} Model F1 Score   = {}\n'.format(Model_Name, np.round(F1_score,3)))
    print('The {} Model Recall     = {}\n'.format(Model_Name, np.round(Recall,3)))
    print('--------------------------------------------------\n')
    
    #Confusion_Matrix = metrics.confusion_matrix(Y_test, Pred_Value)
    #plot_confusion_matrix(Confusion_Matrix,class_names=['Normal', 'Attack'],figsize=(5.55,5), colorbar= "blue")
    plot_roc_curve(Model_Abb, X_test, Y_test)
import mtrices
import validation
