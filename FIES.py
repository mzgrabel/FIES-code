import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import TimeSeriesSplit


FIES = pd.read_excel(r'C:/Users/mzgra/Documents/Research2022/FIES.xlsx')
FIES = FIES.drop(columns=["CareEpi_EndDt", "CareEpi_StartDt", "encounterdate"])
X = FIES.loc[:, FIES.columns != 'FIES_final']
y = FIES['FIES_final']

split = pd.read_csv(r'C:/Users/mzgra/Documents/Research2022/PtSplit.csv')


A = pd.DataFrame()  # training set
for i in range(21616):
        t = FIES['eDWID'] == split['A_develop'][i]
        A = A.append(FIES[t])
    
B = pd.DataFrame()    # validation set
for i in range(21616):
        t = FIES['eDWID'] == split['B_validation'][i]
        B = B.append(FIES[t])
        
C = pd.DataFrame()    # unmasked set
for i in range(21616):
        t = FIES['eDWID'] == split['C_unmasked'][i]
        C = C.append(FIES[t])        

D = pd.DataFrame()    # masked set
for i in range(21616):
        t = FIES['eDWID'] == split['D_masked'][i]
        D = D.append(FIES[t])        

X_train = A.loc[:, FIES.columns != 'FIES_final']
y_train = A['FIES_final']

X_test = B.loc[:, FIES.columns != 'FIES_final']
y_test = B['FIES_final']


# tss = TimeSeriesSplit(n_splits = 2)  # time series split
# train, test = tss.split(FIES)
# train_split = train[1]  # extract train and test splits
# test_split = test[1]
# train_fies = FIES.iloc[train_split, :]
# test_fies = FIES.iloc[test_split, :]

# train_fies = FIES[FIES.eDWID <= 956930676] # based ona cut off point so that sequences are all there
# test_fies = FIES[FIES.eDWID > 956930676]

# X_train = train_fies.iloc[:,train_fies.columns != 'FIES_final'] # extract X and y
# y_train = train_fies['FIES_final']


# X_test = test_fies.iloc[:, test_fies.columns != 'FIES_final']
# y_test = test_fies['FIES_final']

X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

RNN = Sequential()
        
RNN.add(LSTM(units = 50, return_sequences = True, input_shape = (50, 1))) #LSTM input issues
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50, return_sequences=True))
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50, return_sequences=True))
RNN.add(Dropout(0.2))

RNN.add(LSTM(units = 50))
RNN.add(Dropout(0.2))

RNN.add(Dense(units = 1))

RNN.compile(optimizer= 'adam', loss = 'mean_squared_error')

RNN.fit(X_train, y_train, epochs = 20, batch_size = 64) # train

preds = RNN.predict(X_test) 
preds
sum(sum(preds >= 0.719775))
sum(sum(preds < 0.719775))


fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]] 
# 0.719775 best cut off

y_pred = np.zeros(len(preds))
for i in range(len(preds)):
    if preds[[i]][0][0] >= 0.719775:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
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





















# y = FIES["FIES_final"]
# X = FIES.loc[:, FIES.columns != "FIES_final"]
#training_set = X.iloc[:,:].values
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range = (0,1))
# training_set_scaled = sc.fit_transform(training_set)

# train = training_set_scaled.reshape(training_set_scaled.shape[0], 1, training_set_scaled.shape[1])


#rnn = RNN(lstm_size=128, num_layers=1, batch_size=100, learning_rate=0.001)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)

# rnn.train(X_train, y_train, num_epochs=40)

# preds = rnn.predict(X_test)
# y_true = y_test[:len(preds)]
# print('Test Acc.: %.3f' % (np.sum(preds == y_true) / len(y_true)))
    
                    
# test
import statsmodels.api as sm
df = sm.datasets.get_rdataset('weather', 'nycflights13').data
                
df['observation_time'] = pd.to_datetime(df.time_hour)
df.drop(columns=['year', 'month', 'day', 'hour', 'time_hour'], inplace=True)
tss = TimeSeriesSplit(n_splits=2)
train_splits, test_splits = tss.split(df)
train_split = train_splits[0]
test_split = test_splits[0]

train_df = df.iloc[train_split, :]
test_df = df.iloc[test_split, :]

min_date = df.observation_time.min()
max_date = df.observation_time.max()
print("Min:", min_date, "Max:", max_date)

train_percent = .75
time_between = max_date - min_date
train_cutoff = min_date + train_percent*time_between
train_cutoff

train_df = df[df.observation_time <= train_cutoff]
test_df = df[df.observation_time > train_cutoff]
