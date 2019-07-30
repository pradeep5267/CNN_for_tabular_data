#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'pucho_technologies_submit'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# The competition I participated in is Santander Customer Transaction Prediction. The goal of this competiton is to predict or identify
# who will make a transaction.<br>
# This is a binary prediction problem since the goal is to predict whether the customer will make a transaction or not (ie true or false).
#%% [markdown]
# import the necessary libraries

#%%
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,Dropout

import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
# import the dataset, since im using a windows im using double slash notation also directory location will be different for your system

#%%
train_df =  pd.read_csv('./train.csv')

#%% [markdown]
# Extract the target column as y
# Also drop the the target and ID_code from the training dataset since the target labels have to be dropped for training the model and ID_code is just the identification tag and doesnt contribute anything to the training.

#%%
y = train_df['target'].values
train_df.drop(['target','ID_code'],axis=1,inplace=True)

#%% [markdown]
# Check for null value if the null values exist, further digging needs to be done whether to impute the null values or to drop the null values

#%%
train_df.isnull().sum().sum()

#%% [markdown]
# there are no null values in this dataset.
# Moving on, care needs to be taken while dealing with highly correleated column values as it may cause a significant skew in training weights.
# This can be achieved using .corr() command, also using .abs() will return absoulte values in the correlation matrix
# Using seaborn one can plot the correlation matrix to view the highly correlated columns which can be visulaized using a heatmap plot.

#%%
corr_matrix = train_df.corr().abs()


#%%
sns.heatmap(corr_matrix)

#%% [markdown]
# A rather neat looking zero correlation heatmap.
# However since the number of features are not clearly visible (due to display size) a distribution plot of the correlation matrix will 
# show how values are intertwined with each other.

#%%
dist_features = corr_matrix.values.flatten()

sns.distplot(dist_features, color="Red", label="train")

#%% [markdown]
# A sharp spike and nothing else, this proves that the columns in the dataset are uncorrelated with each other.
# Lets extract the significant features from the dataset, this can be achieved using random forest classifiers feature extrator routine

#%%
model = ExtraTreesClassifier()
model.fit(train_df,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

#%% [markdown]
# Plotting the significant features

#%%
feat_importances = pd.Series(model.feature_importances_, index=train_df.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

#%% [markdown]
# Using argsort in descending order and picking out the first 5 elements in the important features list

#%%
feat_imp = model.feature_importances_
feat_imp_desc = np.argsort(feat_importances)[::-1][0:5]
feat_imp_desc


#%%
feature_names = train_df.columns.values


#%%
top = train_df.loc[:, feature_names[feat_imp_desc]]

top['target'] = y

#%% [markdown]
# Plotting a pairplot of the top 5 features and target variable will set a clearer picture of what kind of dataset we're dealing with.
# 
# From the scatter plots of feature variables its easy to observe 2 things:<br>
# 1) All the feature variables are distributed uniformly between their bounding limits. <br>
# 2) A very clear seperation between the target and feature variables that indicates that there are no overlapping features.<br>
# 
# As far as the target variable is concerned:<br>
# 1)The number of 0s out weigh 1s by a significant number ie only ~15 percent of the dataset correspond to people making a transaction.<br>
# <br>Whenever an imbalanced dataset is under consideration, one needs to decide whether to use oversampling to reduce the effects of data imbalance. However such oversampling introduces noise to the dataset which would hinder the models' ability to learn from the dataset.
# This would be accentuated in this dataset which has very a uniform distribution, no overlap or data leaks.
# 

#%%
sns.pairplot(top,diag_kind='hist' )

#%% [markdown]
# The dataset is divided into 180000 for training and the rest for validation<br>
# The dataset was reshaped as 10x10x2 matrix of 180000 samples since it was 180000x200 dataset

#%%
#--Feature selection
features = [X for X in train_df.columns.values.tolist() if X.startswith("var_")]

#--Scaling data and store scaling values
scaler = preprocessing.StandardScaler().fit(train_df[features].values)

X = scaler.transform(train_df[features].values)


#--training & test stratified split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.10)
# reshape dataset
X_train=np.reshape(X_train,(180000,10,10,2))
X_valid=np.reshape(X_valid,(20000,10,10,2))
print(X_train.shape)
print(X_valid.shape)

#%% [markdown]
# 50 epochs were selected since during early stopping method the loss flat lined after 53 epochs
#%% [markdown]
# The input layer has a batch normalization for easier/efficient computation.
# 
# The 1st stage has 128 filters however the second stage has more number of filters to capture more patterns in the dataset.<br>
# 
# The output layer is again a standard dense(Fully connected) layer with sigmoid activation
# 
# The model is compiled for a binary cross entropy loss since its a binary prediction problem.

#%%
#--Model training        
def Convnet(input_shape = (10,10,2),classes = 1):

    X_input = Input(input_shape)
 
    # Stage 1 input
    X = Conv2D(64,kernel_size=(3,3),strides=(1,1),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
    X = Dropout(rate=0.2)(X)
    
    # Stage 2 hidden
    X = Conv2D(128,kernel_size=(2,2),strides=(1,1),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
    X = Dropout(rate=0.1)(X)
    
    X = Conv2D(128,kernel_size=(3,3),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
    X = Dropout(rate=0.1)(X)
    
    X = Conv2D(256,kernel_size=(2,2),strides=(1,1),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
    X = Dropout(rate=0.1)(X)
    
    # Stage 3 output
    X = Conv2D(64,kernel_size=(2,2),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization()(X)
    X = Activation("tanh")(X)
 
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid')(X)
 

    model = Model(inputs=X_input,outputs=X)
 
    return model


#%%
model = Convnet()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_valid, y_valid),verbose = 1)


#%%
#--make prediction
test_df =  pd.read_csv('./test.csv')

X_test = scaler.transform(test_df[features].values)
X_test = np.reshape(X_test,(200000,10,10,2)) 
prediction = model.predict(X_test)


#%%
result = pd.DataFrame({"ID_code": test_df.ID_code.values})
result["target"] = prediction
result.to_csv("predictions.csv", index=False)
model.save('santander_cust_trans_cnn.h5')

#%% [markdown]
# Note:
# First attempt was made using logistic regression which yielded 50% accuracy after submission.<br>
# Second attempt was made using Random Forest classifiers which yielded 55.3% accuracy after submission.<br>
# Third attempt was made using Random Forest classifiers by increasing the depth of the nodes and increasing the leafs which yielded 53.7% accuracy after submission.Such Deep decision trees should yield an overfitting model which was not the case since public test dataset validation loss was still low and validation accuracy was above 65%.<br>
# 
# 
# Improvements:<br>
# 1) If more time had been spent on feature engineering, then there's a high chance that the final model's score can be improved a bit further.
