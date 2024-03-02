import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from emnist import extract_training_samples,extract_test_samples
#mnist =tf.keras.datasets.emnist
(xTrain,yTrain),(xTest,yTest)=extract_training_samples('letters'),extract_test_samples('letters')
xTrain=xTrain/255
xTest=xTest/255

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(62,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xTrain,yTrain,epochs=10)
model.evaluate(xTest,yTest)

model.summary()





