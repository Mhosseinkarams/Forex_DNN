import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load and preprocess your input data
data = pd.read_csv('/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1d_2.csv')
# Preprocess and normalize the data as required

# Define the number of past time steps to consider
num_timesteps = 32

# Create input and output sequences
input_seq = []
output_seq = []
for i in range(len(data) - num_timesteps - 1):
    input_seq.append(data.iloc[i : i + num_timesteps][[ 'Open','High', 'Low','Close' , 'Vol']].values)
    output_seq.append(data.iloc[i + num_timesteps][[ 'Open','High', 'Low','Close' , 'Vol']].values)  # Assuming output is the next data point

# Convert the input and output sequences to numpy arrays
input_seq = np.array(input_seq)
output_seq = np.array(output_seq)

# Split the data into training and testing sets
train_size = int(len(input_seq) * 0.8)
train_input = input_seq[:train_size]
train_output = output_seq[:train_size]
test_input = input_seq[train_size:]
test_output = output_seq[train_size:]

# Build the generator model
generator = Sequential()
generator.add(LSTM(units=128, return_sequences=True, input_shape=(num_timesteps, input_seq.shape[2])))
# Add more LSTM layers or other layers as needed
generator.add(Dense(units=output_seq.shape[1], activation='sigmoid'))

# Build the discriminator model
discriminator = Sequential()
discriminator.add(LSTM(units=128, return_sequences=True, input_shape=(num_timesteps, output_seq.shape[1])))
# Add more LSTM layers or other layers as needed
discriminator.add(Dense(units=1, activation='sigmoid'))

# Compile the generator and discriminator models
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# Combine the generator and discriminator into a GAN model
discriminator.trainable = False
gan_input = keras.Input(shape=(num_timesteps, input_seq.shape[2]))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# Training loop
batch_size = 32
epochs = 100
for epoch in range(epochs):
    for batch in range(len(train_input) // batch_size):
        # Train the discriminator
        real_data = train_output[batch * batch_size : (batch + 1) * batch_size]
        fake_data = generator.predict(train_input[batch * batch_size : (batch + 1) * batch_size])
        discriminator.trainable = True
        discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

        # Train the generator
        discriminator.trainable = False
        gan.train_on_batch(train_input[batch * batch_size : (batch + 1) * batch_size], np.ones((batch_size, 1)))

    # Print the current epoch and loss
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {gan.evaluate(test_input, np.ones((len(test_input), 1)))}")

# Use the trained generator for predictions
predictions = generator.predict(test_input)

# Evaluate the performance of the model
# Add your evaluation metrics for forex trend prediction

# Save the trained models if needed
generator.save("generator_model.h5")
discriminator.save("discriminator_model.h5")
