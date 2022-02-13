import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

data = pd.read_csv('/content/drive/MyDrive/fer2013.csv')    ## write suitable location of file
data ## display data 

data.emotion ## display properties of emotion coloumn

data.emotion.unique()

convert_array = data.pixels.apply(lambda i : np.array(i.split(' ')).reshape(48,48,1).astype('float32'))

convert_array.shape

convert_array = np.stack(convert_array)

convert_array[0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(convert_array,data.emotion.values,test_size=0.2)

X_train=X_train/255.0
X_test = X_test/255.0

plt.imshow(X_train[1].reshape(48,48))

import tensorflow as tf
df_model = tf.keras.models.Sequential([   tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape = (48,48,1)),
                                          tf.keras.layers.MaxPool2D(2,2),
                                          tf.keras.layers.BatchNormalization(),
                                          
                                          tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                          tf.keras.layers.MaxPool2D(2,2),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                          tf.keras.layers.MaxPool2D(2,2),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
                                          tf.keras.layers.MaxPool2D(2,2),
                                          tf.keras.layers.BatchNormalization(),  
                                           
                                          tf.keras.layers.Flatten(),    
                                          tf.keras.layers.Dense(4096,activation='relu'),
                                          tf.keras.layers.Dropout(.3),
                                          tf.keras.layers.Dense(7,activation = 'softmax')                          
                                   ])

df_model.summary()

df_model.compile( 
                  optimizer= 'adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                )

df_model.fit(X_train,y_train, epochs=10)


y_pred = df_model.predict(X_test)

df_model.evaluate(X_test,y_test)

y_predicted = [np.argmax(i) for i in y_pred]


confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted)

from sklearn.metrics import confusion_matrix , classification_report
print("Classification Report: \n", classification_report(y_test, y_predicted))

