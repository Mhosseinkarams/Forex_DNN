#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import time

#Open model
model = tf.keras.models.load_model('forex_classifier_1.h5')

#Open data
data = pd.read_csv('GBPUSD_1h.csv')
data = data.drop(['Datetime'], axis=1)
data = np.array(data)
#Predict
while True:

    pred = model.predict(data[-2:])
    print('--------------------------------------------------')

    print(pred)
    time.sleep(300)