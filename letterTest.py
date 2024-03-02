import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

dataRaw=pd.read_csv("letter-recognition.csv")
dataRaw.head()
dataRaw.tail()
print("Dimensions: ",dataRaw.shape,"\n")
print(dataRaw.info())
dataRaw.head()
let=list(np.sort(dataRaw['letter'].unique()))
print(let)
print(dataRaw.columns)

(xTrain,yTrain),(xTest,yTest)=mnist.load_data()
xTrain=xTrain/255
xTest=xTest/255

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(10,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xTrain,yTrain,epochs=10)
model.evaluate(xTest,yTest)

model.summary()


