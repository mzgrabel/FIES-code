import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import tensorflow as tf
import random

random.seed(10)
# Data ------------------------------------------------------------------------
FIES = pd.read_excel(r'C:/Users/mzgra/Documents/Research2022/FIES.xlsx')
#FIES = FIES.drop(columns=["CareEpi_EndDt", "CareEpi_StartDt", "encounterdate", "year", "zcta", "FEV1_pct_predicted", "alive", "LT", "tSince_BL", "SESlow_ever", "PA_ever", "isOnEnzymes_ever", "Vx770_ever", "Vx809_ever", "smoking_ever", "smoking_household_ever", "second_smoke_ever", "mrsa_ever" ])
X = FIES.loc[:, FIES.columns != 'FIES_final']
y = FIES['FIES_final']

# splitting data from prior paper. These are the groups of eDWIDs used in Szczesniaks paper.
split = pd.read_csv(r'C:/Users/mzgra/Documents/Research2022/PtSplit.csv')


# Extract groups

B = pd.DataFrame()    # validation set
for i in range(21616):
        t = FIES['eDWID'] == split['B_validation'][i]
        B = B.append(FIES[t])
        
C = pd.DataFrame()    # unmasked set / training set
for i in range(21616):
        t = FIES['eDWID'] == split['C_unmasked'][i]
        C = C.append(FIES[t])        

D = pd.DataFrame()    # masked set / test set
for i in range(21616):
        t = FIES['eDWID'] == split['D_masked'][i]
        D = D.append(FIES[t])        

# Table 2
A = pd.concat([C,D]) # Development Cohort
len(np.unique(A['eDWID'])) # unique patients
len(np.unique(B['eDWID']))

len(split['A_develop'])
21616 - 21101 # number of patients in development set - total found

sum(pd.notna(split['B_validation']))
5405 - 5274 # number of patients in validation set - total found

# Development
# Sex
len(np.unique(A['eDWID'][A['Sex'] == 0])) # M
len(np.unique(A['eDWID'][A['Sex'] == 1])) # F

# Age
m = A.groupby('eDWID').mean() # get mean age of visits
np.mean(m['encounterage']) # get mean of all ages of patients
max(m['encounterage'])
min(m['encounterage'])

# Number of Visits
n = A.groupby('eDWID').mean('numVisityr')
np.mean(n['numVisityr'])

# FIES events
f = A.groupby('eDWID').sum('FIES_final')
np.mean(f['FIES_final'])

# Cohorts
len(np.unique(A['eDWID'][A['Birth_cohort'] == 1])) # < 1981
len(np.unique(A['eDWID'][A['Birth_cohort'] == 2])) #  1981 - 1988
len(np.unique(A['eDWID'][A['Birth_cohort'] == 3])) #  1989-1994
len(np.unique(A['eDWID'][A['Birth_cohort'] == 4])) #  1995-1998
len(np.unique(A['eDWID'][A['Birth_cohort'] == 5])) #  1999-2005
len(np.unique(A['eDWID'][A['Birth_cohort'] == 6])) # > 2005

# Validation
# Sex
len(np.unique(B['eDWID'][B['Sex'] == 0])) # M
len(np.unique(B['eDWID'][B['Sex'] == 1])) # F

# Age
mm = B.groupby('eDWID').mean() # get mean age of visits
np.mean(mm['encounterage']) # get mean of all ages of patients
max(mm['encounterage'])
min(mm['encounterage'])

# Number of Visits
nn = B.groupby('eDWID').mean('numVisityr')
np.mean(nn['numVisityr'])

# FIES events
ff = B.groupby('eDWID').sum('FIES_final')
np.mean(ff['FIES_final'])

# Cohorts
len(np.unique(B['eDWID'][B['Birth_cohort'] == 1])) # < 1981
len(np.unique(B['eDWID'][B['Birth_cohort'] == 2])) #  1981 - 1988
len(np.unique(B['eDWID'][B['Birth_cohort'] == 3])) #  1989-1994
len(np.unique(B['eDWID'][B['Birth_cohort'] == 4])) #  1995-1998
len(np.unique(B['eDWID'][B['Birth_cohort'] == 5])) #  1999-2005
len(np.unique(B['eDWID'][B['Birth_cohort'] == 6])) # > 2005



C = C.drop(columns=['eDWID'])
X_train = C.loc[:, C.columns != 'FIES_final']
y_train = C['FIES_final']
D = D.drop(columns=['eDWID'])
X_test = D.loc[:, D.columns != 'FIES_final']
y_test = D['FIES_final']

# -----------------------------------------------------------------------------

# Model evaluations -----------------------------------------------------------
def Evaluate(model, y_pred):
    def misclassificationrate(cm):
        error = cm[0,1] + cm[1,0]
        total = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
        mr = error / total * 100
        return round(mr, 2)
    
    def Sensitivity(cm): # Sensitivity TP/(TP+FN)
        sen = cm[0,0]/(cm[0,0]+cm[0,1])
        return sen
    
    def Specificity(cm): # Specificity TN/(TN+FP)
        spe = cm[1,1]/(cm[0,1]+cm[1,1])
        return spe
    
    # ROC Curve plot
    fig, ax = plt.subplots()
    y_pred_proba = model.predict_proba(X_test)[::,1] # get probabilities
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba) # get AUC
    plt.plot(fpr,tpr,label="FIES data, auc="+str(auc))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()
        
    cm = metrics.confusion_matrix(y_test, y_pred) # Confusion matrix of test data vs predicted data
    cm
                
    sen = Sensitivity(cm)
    spe = Specificity(cm)
        
    # (optional) create plot of confusion matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
          
    # print essential validation values
    print('AUC', auc)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Sensitivity", sen)
    print('Specificity', spe)
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

def ANNEvaluate(model, y_pred):
    def misclassificationrate(cm):
        error = cm[0,1] + cm[1,0]
        total = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
        mr = error / total * 100
        return round(mr, 2)
    
    def Sensitivity(cm):
        sen = cm[0,0]/(cm[0,0]+cm[0,1])
        return sen
    
    def Specificity(cm):
        spe = cm[1,1]/(cm[0,1]+cm[1,1])
        return spe
    
    # ROC Curve plot
    fig, ax = plt.subplots()
    y_pred_proba = y_pred[::,1] #neural networks output the probabilities
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)  # AUC
    plt.plot(fpr,tpr,label="FIES data, auc="+str(auc))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()

    y_pred = np.argmax(y_pred, axis = 1)
        
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds']
    

    cm = metrics.confusion_matrix(y_test, y_pred)
    cm        
    
    misclassificationrate(cm)
    sen = Sensitivity(cm)
    spe = Specificity(cm)
    
    # (optional) Confusion matrix plot
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
                
    print('AUC', auc)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Sensitivity", sen)
    print('Specificity', spe)
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    


# -----------------------------------------------------------------------------


# Logistic Regression
lr = LogisticRegression(C = 100.0, random_state = 1)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
Evaluate(lr, y_pred_lr) # call evaluate function on these predictions

# Random Forest
forest = RandomForestClassifier(criterion='gini', n_estimators = 25, random_state=1, n_jobs = 2)
forest.fit(X_train, y_train)

y_pred_rf = forest.predict(X_test)
Evaluate(forest, y_pred_rf) # call evaluate function on these predictions


# KNN
knn = KNeighborsClassifier(n_neighbors = 2, p = 2, metric = 'minkowski')
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
Evaluate(knn, y_pred_knn) # call evaluate function on these predictions


# ANN
y_train_onehot = tf.keras.utils.to_categorical(y_train)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=50, input_dim=X_train.shape[1], kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros', activation = 'tanh'))
ann.add(tf.keras.layers.Dense(units = 50, input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation = 'tanh')) 
ann.add(tf.keras.layers.Dense(units = y_train_onehot.shape[1], input_dim = 50, kernel_initializer = 'glorot_uniform', bias_initializer='zeros', activation = 'softmax'))

sgd_optimizer = tf.keras.optimizers.SGD(lr = 0.001, decay = 1e-7, momentum = 0.9)

ann.compile(optimizer = sgd_optimizer, loss = 'categorical_crossentropy')

ann.fit(X_train, y_train_onehot, batch_size = 64, epochs = 25, verbose=1, validation_split = 0.1)
y_pred_ann = ann.predict(X_test, verbose = 0)

ANNEvaluate(ann, y_pred_ann) # call evaluate function on these predictions

# RNN -------------------------------------------------------------------------

X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1) # reshape train data to 3 dim input

RNN = Sequential()
        
RNN.add(LSTM(units = 50, return_sequences = True, input_shape = (34, 1))) 
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50, return_sequences=True))
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50, return_sequences=True))
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50))
RNN.add(Dropout(0.2))

RNN.add(Dense(units = 1))

RNN.compile(optimizer= 'adam', loss = 'mean_squared_error')

RNN.fit(X_train, y_train, epochs = 25, batch_size = 64) # train

preds = RNN.predict(X_test) 
preds

# find best cut off point of probability for 0 or 1
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds']


y_pred = np.zeros(len(preds))
for i in range(len(preds)):
    if preds[[i]][0][0] >= 0.628183: # may have to change the numeric value based on the threshold value above
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
# Validation

cm = metrics.confusion_matrix(y_test, y_pred)
    
def misclassificationrate(cm):
    error = cm[0,1] + cm[1,0]
    total = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
    mr = error / total * 100
    return round(mr, 2)
    
def Sensitivity(cm):
    sen = cm[0,0]/(cm[0,0]+cm[0,1])
    return sen
    
def Specificity(cm):
    spe = cm[1,1]/(cm[0,1]+cm[1,1])
    return spe
    
mr = misclassificationrate(cm)
sen = Sensitivity(cm)
spe = Specificity(cm)
mr
sen
spe

auc = metrics.roc_auc_score(y_test, preds[:,0][:][:][:])
auc

# ROC Curve plot
fig, ax = plt.subplots()
fpr, tpr, _ = metrics.roc_curve(y_test,  preds[:,0][:][:][:])
plt.plot(fpr,tpr,label="FIES data, auc="+str(auc))
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()

print('AUC', auc)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Sensitivity", sen)
print('Specificity', spe)
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# Figure 1A

def plotmetrics(model, y_pred):
    if model == ann:
        y_pred_proba = y_pred[::,1]
    elif model == RNN:
        y_pred_proba = y_pred[:,0][:][:][:]
    else:
        y_pred_proba = model.predict_proba(X_test)[::,1] # get probabilities

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    return fpr, tpr

fprlr, tprlr = plotmetrics(lr, y_pred_rf)
fprrf, tprrf = plotmetrics(forest, y_pred_rf)
fprk, tprk = plotmetrics(knn, y_pred_knn)
fpra, tpra = plotmetrics(ann, y_pred_ann)
fprr, tprr = plotmetrics(RNN, preds)


plt.plot(fprlr, tprlr, color = 'red', label = 'Logistic Regression')
plt.plot(fprrf, tprrf, color = 'green', label = 'Random Forest')
plt.plot(fprk, tprk, color = 'orange', label = 'KNN')
plt.plot(fpra, tpra, color = 'blue', label = 'ANN')
plt.plot(fprr, tprr, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()


# External Cross Validation ---------------------------------------------------
from sklearn.model_selection import cross_val_predict

B = B.drop(columns=['eDWID'])
# Use set B for external cross validation
XB = B.loc[:, B.columns != 'FIES_final']
yB = B['FIES_final']


# Cross validation on set Bs predictions vs the true values in set B
def CrossValExt(pred):        
    cm = metrics.confusion_matrix(yB, pred)
    auc = metrics.roc_auc_score(yB, pred)
    sen = Sensitivity(cm)
    spe = Specificity(cm)
    
    # ROC Curve plot
    fig, ax = plt.subplots()
    fpr, tpr, _ = metrics.roc_curve(yB,  pred)
    plt.plot(fpr,tpr,label="FIES data, auc="+str(auc))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()
    
    print('AUC', auc)
    print("Accuracy:",metrics.accuracy_score(yB, pred))
    print("Sensitivity", sen)
    print('Specificity', spe)
    print("Precision:",metrics.precision_score(yB, pred))
    print("Recall:",metrics.recall_score(yB, pred))

# Logistic Regression

le = cross_val_predict(lr, XB, yB)
CrossValExt(le)
# Random Forest

fe = cross_val_predict(forest, XB, yB)
CrossValExt(fe)
# KNN

ke = cross_val_predict(knn, XB, yB)
CrossValExt(ke)

# ANN

yae = ann.predict(XB)

# get best cut off value from probabilities 
fpr, tpr, thresholds = metrics.roc_curve(yB, yae[:,0])
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

y_pred_a = np.zeros(len(yae[:,0]))
for i in range(len(yae[:,0])):
    if yae[:,0][i] >= 0.343799: # may have to adjust value based on above threshold value
        y_pred_a[i] = 1
    else:
        y_pred_a[i] = 0
        
CrossValExt(y_pred_a)

# RNN

yre = RNN.predict(XB)

# get best cut off value from probabilities 
fpr, tpr, thresholds = metrics.roc_curve(yB, yre[:,0])
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

y_pred_r = np.zeros(len(yre[:,0]))
for i in range(len(yre[:,0])):
    if yre[:,0][i] >= 0.621361: # may have to adjust value based on above threshold value
        y_pred_r[i] = 1
    else:
        y_pred_r[i] = 0
        
CrossValExt(y_pred_r)


# Figure 1B

def plotmetricsex(pred):
    fpr, tpr, _ = metrics.roc_curve(yB,  pred)
    return fpr, tpr

fprlre, tprlre = plotmetricsex(le)
fprrfe, tprrfe = plotmetricsex(fe)
fprke, tprke = plotmetricsex(ke)
fprae, tprae = plotmetricsex(y_pred_a)
fprre, tprre = plotmetricsex(y_pred_r)

plt.plot(fprlre, tprlre, color = 'red', label = 'Logistic Regression')
plt.plot(fprrfe, tprrfe, color = 'green', label = 'Random Forest')
plt.plot(fprke, tprke, color = 'orange', label = 'KNN')
plt.plot(fprae, tprae, color = 'blue', label = 'ANN')
plt.plot(fprre, tprre, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()


# combined Figures

plt.subplot(1,2,1)
plt.plot(fprlr, tprlr, color = 'red', label = 'Logistic Regression')
plt.plot(fprrf, tprrf, color = 'green', label = 'Random Forest')
plt.plot(fprk, tprk, color = 'orange', label = 'KNN')
plt.plot(fpra, tpra, color = 'blue', label = 'ANN')
plt.plot(fprr, tprr, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()

plt.subplot(1,2,2)
plt.plot(fprlre, tprlre, color = 'red', label = 'Logistic Regression')
plt.plot(fprrfe, tprrfe, color = 'green', label = 'Random Forest')
plt.plot(fprke, tprke, color = 'orange', label = 'KNN')
plt.plot(fprae, tprae, color = 'blue', label = 'ANN')
plt.plot(fprre, tprre, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()

