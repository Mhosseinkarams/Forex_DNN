#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Normalization, Dense, Dropout
from tensorflow.keras.models import Sequential
import keras_tuner as kt
import tensorflow_addons as tfa

#Read the data
data = pd.read_csv("GBPUSD_1h_preprocessed.csv")
# data = data.drop(['Datetime'], axis=1)
#Define sets of 100 days data
# x = []
# y = []
# for i in range(len(data)-101):
#     if (i+101<=len(data)-101): 
#         x.append(np.array(data.iloc[i:i+100]))
#         y.append(data.iloc[i+100]['Classification'])
# # Convert the training data to numpy arrays
# x = np.array(x)
# y = np.array(y)
x = data[['Open' , 'High' , 'Low' , 'Close' , 'Volume']]
y = data['Classification']
#Split the data to train, test and validation sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,)

print('x shape =',X_train.shape,'y shape =',y_train.shape)

#Define the model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
#     model.add(layers.Dense(64,activation="relu"
#             ))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", 
                             min_value=32, 
                             max_value=256, 
                             step=32),
                activation="relu"),
            )
        if hp.Boolean(f"norm_{i}"):
            model.add(Normalization())
        
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="softmax"))
    learning_rate = hp.Float("lr", 
                             min_value=1e-4, 
                             max_value=1e-2, 
                             sampling="log")
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_hinge', 
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(hypermodel=build_model,
                        objective="val_loss",
                        max_trials=20,
                        executions_per_trial=1,
                        overwrite=True,
                        directory='output',
                        project_name='my_model')
tuner.search(X_train, 
             y_train,
             validation_data=(X_val, y_val),
             batch_size=64, 
             epochs=300,
             workers=10,
             use_multiprocessing=True)
# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(1,5))
best_model.summary()

# Evaluate the model on the validation set
eval_metrics = best_model.evaluate(X_val, y_val)

# Print the evaluation metrics
print('Evaluation metrics:', best_model.metrics_names)

#Saving the best model
best_model.save('forex_classifier_1.h5')