import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from emnist import extract_training_samples,extract_test_samples

(xTrain,yTrain),(xTest,yTest)=extract_training_samples('balanced'),extract_test_samples('balanced')
#if you do digits it has 99.69 accuracy with 10 out and 64 dense
xTrain=xTrain/255
xTest=xTest/255

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
len(LABELS)
def plotImage(images,predicted_labels,actual_labels):
    numI=len(images)
    numR=3
    numC=3
    randomI=np.random.choice(numI,size=numR*numC,replace=False)
    plt.figure(figsize=(15,5*numR))
    for i,idx in enumerate(randomI):
        plt.subplot(3,3,i+1)
        plt.imshow(images[idx],cmap='gray')
        predictedC=LABELS[predicted_labels[idx]]
        actualC=LABELS[actual_labels[idx]]
        plt.title(f'Predicted:{predictedC}\nAcutal: {actualC}')
        plt.axis('off')
    plt.show()

model=tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,(5,5),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
   # tf.keras.layers.Dropout(.5),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
   # tf.keras.layers.Dropout(.5),
  #  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
   # tf.keras.layers.Dropout(.5),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(47,activation='softmax')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xTrain,yTrain,epochs=10)
model.evaluate(xTest,yTest)

model.summary()

predictions=model.predict(xTest)
predicted_labels=np.argmax(predictions,axis=1)
plotImage(xTest,predicted_labels,yTest)


