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
import os

os.chdir("C://Users//mzgra//OneDrive//Research2022//Code")

random.seed(10)
# Data ------------------------------------------------------------------------

B = pd.read_csv(r'oos_validation_data.csv')  # validation set
B = B.drop(B.iloc[:,36:68],axis = 1)

C = pd.read_csv(r'development_data.csv') # unmasked set / training set
C = C.drop(C.iloc[:,36:68],axis = 1)

D = pd.read_csv(r'masked_forecasting_data.csv')  # masked set / test set
D = D.drop(D.iloc[:,36:68],axis = 1)

C = C.drop(columns=['eDWID'])
X_train = C.loc[:, C.columns != 'FIES']
y_train = C['FIES']
D = D.drop(columns=['eDWID'])
X_test = D.loc[:, D.columns != 'FIES']
y_test = D['FIES']

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
        spe = cm[1,1]/(cm[1,0]+cm[1,1])
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

# def NNEvaluate(model, y_pred):
#     def misclassificationrate(cm):
#         error = cm[0,1] + cm[1,0]
#         total = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
#         mr = error / total * 100
#         return round(mr, 2)
    
#     def Sensitivity(cm):
#         sen = cm[0,0]/(cm[0,0]+cm[0,1])
#         return sen
    
#     def Specificity(cm):
#         spe = cm[1,1]/(cm[0,1]+cm[1,1])
#         return spe
    
#     def OptimalCutoff(pred): # since NN predictions are probabilities we have to find the best cut off point to make those hard predictions
#         fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
#         i = np.arange(len(tpr)) # index for df
#         roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
#         roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        
#         return list(roc_t['thresholds'])

#     c = OptimalCutoff(y_pred) 

#     y_pred_hard = np.zeros(len(y_pred))
#     for i in range(len(y_pred)):
#         if y_pred[[i]][0][0] >= c: # use hard cut off point to make prediction 1 if its larger or 0 if smaller
#             y_pred_hard[i] = 1
#         else:
#             y_pred_hard[i] = 0
            
#     # ROC Curve plot
#     fig, ax = plt.subplots()
#     fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
#     auc = metrics.roc_auc_score(y_test, y_pred)  # AUC
#     plt.plot(fpr,tpr,label="FIES data, auc="+str(auc))
#     plt.xlabel('1-Specificity')
#     plt.ylabel('Sensitivity')
#     plt.title('ROC Curve')
#     plt.legend(loc=4)
#     plt.show()

#     cm = metrics.confusion_matrix(y_test, y_pred_hard)
#     cm        
    
#     misclassificationrate(cm)
#     sen = Sensitivity(cm)
#     spe = Specificity(cm)
    
#     # (optional) Confusion matrix plot
#     class_names=[0,1] # name  of classes
#     fig, ax = plt.subplots()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names)
#     plt.yticks(tick_marks, class_names)
#     # create heatmap
#     sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
#     ax.xaxis.set_label_position("top")
#     plt.tight_layout()
#     plt.title('Confusion matrix', y=1.1)
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label') 
                
#     print('AUC', auc)
#     print("Accuracy:",metrics.accuracy_score(y_test, y_pred_hard))
#     print("Sensitivity", sen)
#     print('Specificity', spe)
#     print("Precision:",metrics.precision_score(y_test, y_pred_hard))
#     print("Recall:",metrics.recall_score(y_test, y_pred_hard))
    
#     return y_pred_hard.astype(int)


# -----------------------------------------------------------------------------


# Logistic Regression
lr = LogisticRegression(C = 100.0, random_state = 1)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
Evaluate(lr, y_pred_lr) # call evaluate function on these predictions

y_pred_proba_lr = lr.predict_proba(X_test)[::,1] # get probabilities

ppdevlr = pd.DataFrame(y_pred_proba_lr)
# ppdevlr.to_excel("PredictedProbabilitiesDevLR.xlsx")

# Random Forest
forest = RandomForestClassifier(criterion='gini', n_estimators = 25, random_state=1, n_jobs = 2)
forest.fit(X_train, y_train)

y_pred_rf = forest.predict(X_test)
Evaluate(forest, y_pred_rf) # call evaluate function on these predictions

y_pred_proba_rf = forest.predict_proba(X_test)[::,1] # get probabilities

ppdevrf = pd.DataFrame(y_pred_proba_rf)
# ppdevrf.to_excel("PredictedProbabilitiesDevRF.xlsx")

# KNN
knn = KNeighborsClassifier(n_neighbors = 2, p = 2, metric = 'minkowski')
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
Evaluate(knn, y_pred_knn) # call evaluate function on these predictions

# ANN

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train_ann = sc.fit_transform(X_train)
# X_test_ann = sc.fit_transform(X_test)

# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=50, input_dim=X_train.shape[1], activation = 'tanh')) # kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros',
# ann.add(tf.keras.layers.Dense(units = 50, activation = 'tanh')) #input_dim=50, 
# ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# ann.fit(X_train, y_train, batch_size = 64, epochs = 25) #, verbose=1, validation_split = 0.1
# y_ppred_ann = ann.predict(X_test, verbose = 0)

# yya = NNEvaluate(ann, y_ppred_ann) # call evaluate function on these predictions

# RNN -------------------------------------------------------------------------

X_train_rnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1) # reshape train data to 3 dim input

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

RNN.fit(X_train_rnn, y_train, epochs = 25, batch_size = 64) # train

preds = RNN.predict(X_test) 
preds


ppdevrnn = pd.DataFrame(preds)
# ppdevrnn.to_excel("PredictedProbabilitiesDevRNN.xlsx")



# find best cut off point of probability for 0 or 1
def OptimalCutoff(pred): # since NN predictions are probabilities we have to find the best cut off point to make those hard predictions
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        
    return list(roc_t['thresholds'])

c = OptimalCutoff(preds)

y_pred = np.zeros(len(preds))
for i in range(len(preds)):
    if preds[[i]][0][0] >= c:
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
    spe = cm[1,1]/(cm[1,0]+cm[1,1])
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
    # if model == ann:
    #     y_pred_proba =  y_pred[:,0][:][:][:]
    if model == RNN:
        y_pred_proba = y_pred[:,0][:][:][:]
    else:
        y_pred_proba = model.predict_proba(X_test)[::,1] # get probabilities

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    return fpr, tpr

fprlr, tprlr = plotmetrics(lr, y_pred_lr)
fprrf, tprrf = plotmetrics(forest, y_pred_rf)
fprk, tprk = plotmetrics(knn, y_pred_knn)

# fpra, tpra = plotmetrics(ann, y_ppred_ann)

fprr, tprr = plotmetrics(RNN, preds)


plt.plot(fprlr, tprlr, color = 'red', label = 'Logistic Regression')
plt.plot(fprrf, tprrf, color = 'green', label = 'Random Forest')
plt.plot(fprk, tprk, color = 'orange', label = 'KNN')
#plt.plot(fpra, tpra, color = 'blue', label = 'ANN')
plt.plot(fprr, tprr, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()

Xlr = pd.DataFrame(fprlr, columns = ['Xlr'])
Ylr = pd.DataFrame(tprlr, columns = ['Ylr'])

plotmetrics_lr = pd.concat([Xlr, Ylr], axis= 1)

Xrf = pd.DataFrame(fprrf, columns = ['Xrf'])
Yrf = pd.DataFrame(tprrf, columns = ['Yrf'])

plotmetrics_rf = pd.concat([Xrf, Yrf], axis = 1)

Xk = pd.DataFrame(fprk, columns = ['Xk'])
Yk = pd.DataFrame(tprk, columns = ['Yk'])

plotmetrics_k = pd.concat([Xk, Yk], axis = 1)

# Xa = pd.DataFrame(fpra, columns = ['Xa'])
# Ya = pd.DataFrame(tpra, columns = ['Ya'])

# a = pd.concat([Xa, Ya], axis = 1)

Xr = pd.DataFrame(fprr, columns = ['Xr'])
Yr = pd.DataFrame(tprr, columns = ['Yr'])

plotmetrics_r = pd.concat([Xr, Yr], axis = 1)

# plotmetrics_lr.to_excel("DevelopmentplotmetricsLR.xlsx")

# plotmetrics_rf.to_excel("DevelopmentplotmetricsRF.xlsx")

# plotmetrics_k.to_excel("DevelopmentplotmetricsKNN.xlsx")

# # plotmetrics_a.to_excel("DevelopmentplotmetricsANN.xlsx")

# plotmetrics_r.to_excel("DevelopmentplotmetricsRNN.xlsx")




# External Cross Validation ---------------------------------------------------
from sklearn.model_selection import cross_val_predict

B = B.drop(columns=['eDWID'])
# Use set B for external cross validation
XB = B.loc[:, B.columns != 'FIES']
yB = B['FIES']


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
# XBt = sc.fit_transform(XB)
# yae = ann.predict(XB)

def OptimalCutoffANN(pred): # since NN predictions are probabilities we have to find the best cut off point to make those hard predictions
    fpr, tpr, thresholds = metrics.roc_curve(yB, pred)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        
    return list(roc_t['thresholds'])

# c = OptimalCutoffANN(yae)

# y_pred_a = np.zeros(len(yae[:,0]))
# for i in range(len(yae[:,0])):
#     if yae[:,0][i] >= c: 
#         y_pred_a[i] = 1
#     else:
#         y_pred_a[i] = 0
        
# CrossValExt(y_pred_a)

# RNN

yre = RNN.predict(XB)

# get best cut off value from probabilities 
fpr, tpr, thresholds = metrics.roc_curve(yB, yre[:,0])
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

d = OptimalCutoffANN(yre[:,0])

y_pred_r = np.zeros(len(yre[:,0]))
for i in range(len(yre[:,0])):
    if yre[:,0][i] >= d: # may have to adjust value based on above threshold value
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
#fprae, tprae = plotmetricsex(y_pred_a)
fprre, tprre = plotmetricsex(y_pred_r)

plt.plot(fprlre, tprlre, color = 'red', label = 'Logistic Regression')
plt.plot(fprrfe, tprrfe, color = 'green', label = 'Random Forest')
plt.plot(fprke, tprke, color = 'orange', label = 'KNN')
#plt.plot(fprae, tprae, color = 'blue', label = 'ANN')
plt.plot(fprre, tprre, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()


Xlre = pd.DataFrame(fprlre, columns = ['Xlr'])
Ylre = pd.DataFrame(tprlre, columns = ['Ylr'])

plotmetrics_lre = pd.concat([Xlre, Ylre], axis= 1)

Xrfe = pd.DataFrame(fprrfe, columns = ['Xrf'])
Yrfe = pd.DataFrame(tprrfe, columns = ['Yrf'])

plotmetrics_rfe = pd.concat([Xrfe, Yrfe], axis = 1)

Xke = pd.DataFrame(fprke, columns = ['Xk'])
Yke = pd.DataFrame(tprke, columns = ['Yk'])

plotmetrics_kne = pd.concat([Xke, Yke], axis = 1)

# Xae = pd.DataFrame(fprae, columns = ['Xa'])
# Yae = pd.DataFrame(tprae, columns = ['Ya'])

# ae = pd.concat([Xae, Yae], axis = 1)

Xre = pd.DataFrame(fprre, columns = ['Xr'])
Yre = pd.DataFrame(tprre, columns = ['Yr'])

plotmetrics_re = pd.concat([Xre, Yre], axis = 1)

# plotmetrics_lre.to_excel("ValidationplotmetricsLR.xlsx")

# plotmetrics_rfe.to_excel("ValidationplotmetricsRF.xlsx")

# plotmetrics_kne.to_excel("ValidationplotmetricsKNN.xlsx")

# # ae.to_excel("ValidationplotmetricsANN.xlsx")

# plotmetrics_re.to_excel("ValidationplotmetricsRNN.xlsx")





# combined Figures

plt.subplot(1,2,1)
plt.plot(fprlr, tprlr, color = 'red', label = 'Logistic Regression')
plt.plot(fprrf, tprrf, color = 'green', label = 'Random Forest')
plt.plot(fprk, tprk, color = 'orange', label = 'KNN')
#plt.plot(fpra, tpra, color = 'blue', label = 'ANN')
plt.plot(fprr, tprr, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()

plt.subplot(1,2,2)
plt.plot(fprlre, tprlre, color = 'red', label = 'Logistic Regression')
plt.plot(fprrfe, tprrfe, color = 'green', label = 'Random Forest')
plt.plot(fprke, tprke, color = 'orange', label = 'KNN')
#plt.plot(fprae, tprae, color = 'blue', label = 'ANN')
plt.plot(fprre, tprre, color = 'black', label = 'RNN')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves')
plt.legend()
plt.show()

