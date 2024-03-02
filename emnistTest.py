import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from emnist import extract_training_samples,extract_test_samples
#mnist =tf.keras.datasets.emnist
(xTrain,yTrain),(xTest,yTest)=extract_training_samples('letters'),extract_test_samples('letters')
xTrain=xTrain/255
xTest=xTest/255
asciiOf=64
def plotImage(images,predicted_labels,actual_labels):
    numI=len(images)
    numR=3
    numC=3
    randomI=np.random.choice(numI,size=numR*numC,replace=False)
    plt.figure(figsize=(15,5*numR))
    for i,idx in enumerate(randomI):
        plt.subplot(3,3,i+1)
        plt.imshow(images[idx],cmap='gray')
        predictedC=chr(predicted_labels[idx]+asciiOf)
        actualC=chr(actual_labels[idx]+asciiOf)
        plt.title(f'Predicted:{predictedC}\nAcutal: {actualC}')
        plt.axis('off')
    plt.show()

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(62,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xTrain,yTrain,epochs=10)
model.evaluate(xTest,yTest)

model.summary()

predictions=model.predict(xTest)
predicted_labels=np.argmax(predictions,axis=1)
plotImage(xTest,predicted_labels,yTest)

#predicted_char[chr(i+asciiOf) for i in predicted_labels]
#actual_char[chr(i+asciiOf) for i in yTest]
#for pred, actual in zip(precited_char, actual_char):
 #   print(f"Predicted: {pred}, Actual: {actual}")




