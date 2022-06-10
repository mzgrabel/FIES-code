# FIES-code

R code was run on version 1.3.1093 
Python code was run on Spyder IDE 5.1.5 Python version 3.8.12 

R code preprocesses the data 
-uploads raw data
-changes gender to numeric sex variable
-changes birthcohort variable to numeric for each of the cohorts 1 - <1981; 2 - 1981-1988; 3 - 1989-1994; 4 - 1995-1998; 5 - 1999-2005
-looks only for complete cases
-exports to excel file

Python Code runs the models
- uploads the excel file and drops the non numeric date variables
- uploads the data split used in Szczesniak et al. (2019)
- matches the split data IDs to the the processed data into groups B, C, D one for validation, training, and testing 
- Sets the training and test data 
- has evaluation code for the models
- gets the confusion matrix from the predictions and test data 
- finds specificity and sensitivity AUC accuracy precision and recall
- plots the AUC curve
- runs the logisitic regression model from sklearn.linear_model import LogisticRegression package
- fits, predicts, and evaluates the model
- repeats this for random forest and KNN using sklearn.ensemble import RandomForestClassifier and sklearn.neighbors import KNeighborsClassifier
- creates an artifial neural network using tensorflow keras Sequential object with Dense layers using tanh as activation function
- optimizes and compiles the model under categorical crossentropy 
- fits the model  with 25 epochs and a batchsize of 64
- Evaluates the model
- All these models do not account for repeated measures
- Develops an RNN to account for them
- Reshapes the training data for input into RNN
- input shape is the 50 variables with variable amount of repeated measures
- 4 LSTM units and a dense unit
- Fits the model with a batchsize of 64 and 25 epochs 
- gets hard prediction 
- finds best cut off value based on the ROC curve
- gets soft prediction based on that cut off value
- evaluates model as before

Uses dataset B for external cross validation to validate the model on data it hasnt been trained on or tested
- uses from sklearn.model_selection import cross_val_predict to predict the models on the new data and uses the usual 
evaluation and AUC for logisitic regression, random forest, and KNN
- the neural networks required a more manual method
- create a new prediction on the model 
- find best cut off value and hard prediction
- evaulate that hard prediction vs the true values of that data
- 
