# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
<vul/>dataset = pd.read_csv('Churn_Modelling.csv')</vul>
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
<vul/>onehotencoder = OneHotEncoder(categorical_features = [1])</vul>
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
<vul/>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)</vul>

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
<vul/>classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))</vul>

# Adding the second hidden layer
<vul/>classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))</vul>

# Adding the output layer
<vul/>classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))</vul>

# Compiling the ANN
<vul/>classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])</vul>

# Fitting the ANN to the Training set
<vul/>classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)</vul>

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)