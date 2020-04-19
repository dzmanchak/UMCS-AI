# -*- coding: utf-8 -*-
"""
Spyder Editor

@author Dzmitry Manchak
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('iris.csv')
dataset.head()

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

sc = StandardScaler()
X2 = sc.fit_transform(X) 

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values
 
model = Sequential()
model.add(Dense(30, activation='relu',input_dim=4, kernel_initializer = 'random_normal'))
model.add(Dense(10,activation='relu',kernel_initializer = 'random_normal'))
model.add(Dense(5,activation='relu',kernel_initializer = 'random_normal'))
model.add(Dense(3,activation='softmax',kernel_initializer = 'random_normal'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(X2,Y, batch_size=1, epochs =1000)

eval_model = model.evaluate(X2,Y)
print(eval_model)
s = model.predict(X2)
print(s)




