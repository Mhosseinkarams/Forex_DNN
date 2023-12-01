import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load and preprocess your input data
data = pd.read_csv('/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1d_2.csv')
features = data[['Open', 'High', 'Low', 'Close', 'Vol']].values

# Normalize the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Define the sequence length for LSTM input
sequence_length = 24  # You can adjust this value depending on your data

# Create sequences of input and output data for LSTM training
sequences = []
next_closes = []
for i in range(len(scaled_features) - sequence_length - 1):
    seq = scaled_features[i:i + sequence_length]
    target = scaled_features[i + sequence_length + 1][3]  # Predicting the 'Close' price
    sequences.append(seq)
    next_closes.append(target)

# Convert the sequences and next_closes to numpy arrays
X = np.array(sequences)
y = np.array(next_closes)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(sequence_length, features.shape[1])))
lstm_model.add(Dense(1, activation='linear'))

# Compile the LSTM model
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Build the GAN generator
latent_dim = 100  # Dimension of the GAN latent space

generator = Sequential()
generator.add(Dense(128, input_dim=latent_dim))
generator.add(Dense(sequence_length * features.shape[1], activation='tanh'))
generator.add(Reshape((sequence_length, features.shape[1])))

# Build the GAN discriminator
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(sequence_length, features.shape[1])))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Build the GAN model
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Compile the GAN model
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Function to train the GAN
def train_gan(epochs=1, batch_size=128):
    for e in range(epochs):
        for _ in range(len(X_train) // batch_size):
            # Train the discriminator
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_data = generator.predict(noise)
            real_data = X_train[np.random.randint(0, len(X_train), size=batch_size)]

            X_batch = np.concatenate([real_data, generated_data])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # One-sided label smoothing for the discriminator

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X_batch, y_dis)

            # Train the generator (combined with the discriminator)
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        print(f"Epoch {e + 1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

# Train the GAN
train_gan(epochs=50, batch_size=64)

# Function to generate synthetic data using the GAN generator
def generate_synthetic_data(num_samples):
    noise = np.random.normal(0, 1, size=[num_samples, latent_dim])
    generated_data = generator.predict(noise)
    return generated_data

# Generate synthetic data
synthetic_data = generate_synthetic_data(len(X_test))

# Reshape the synthetic_data to 2D array before inverse_transform
synthetic_data_2d = synthetic_data.reshape(-1, features.shape[1])

# Denormalize the data using the MinMaxScaler's inverse_transform method
synthetic_data_denormalized = scaler.inverse_transform(synthetic_data_2d)

# Reshape it back to the original shape
synthetic_data_denormalized = synthetic_data_denormalized.reshape(synthetic_data.shape[0], sequence_length, features.shape[1])

# Predict using the LSTM model on real and synthetic data
real_predictions = lstm_model.predict(X_test)
synthetic_predictions = lstm_model.predict(synthetic_data_denormalized)

# Reshape real_predictions and synthetic_predictions to (num_samples,)
real_predictions = real_predictions.reshape(-1)
synthetic_predictions = synthetic_predictions.reshape(-1)

# Calculate the min and max values of the original 'Close' prices
close_min = data['Close'].min()
close_max = data['Close'].max()

# Denormalize the predictions manually
real_predictions_denormalized = real_predictions * (close_max - close_min) + close_min
synthetic_predictions_denormalized = synthetic_predictions * (close_max - close_min) + close_min

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get the minimum and maximum values of the original 'Close' prices
close_min = data['Close'].min()
close_max = data['Close'].max()

# Evaluate the LSTM model on real data
actual_close_prices = y_test * (close_max - close_min) + close_min

mse_real = mean_squared_error(actual_close_prices, real_predictions_denormalized)
mae_real = mean_absolute_error(actual_close_prices, real_predictions_denormalized)
rmse_real = np.sqrt(mse_real)

print(f"LSTM Model Evaluation on Real Data:")
print(f"Mean Squared Error (MSE): {mse_real:.4f}")
print(f"Mean Absolute Error (MAE): {mae_real:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_real:.4f}")

# Evaluate the LSTM model on synthetic data
mse_synthetic = mean_squared_error(actual_close_prices, synthetic_predictions_denormalized)
mae_synthetic = mean_absolute_error(actual_close_prices, synthetic_predictions_denormalized)
rmse_synthetic = np.sqrt(mse_synthetic)

print("\nLSTM Model Evaluation on Synthetic Data:")
print(f"Mean Squared Error (MSE): {mse_synthetic:.4f}")
print(f"Mean Absolute Error (MAE): {mae_synthetic:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_synthetic:.4f}")

# Generate some random input sequences for prediction
num_sequences_to_predict = 500
random_indices = np.random.randint(0, len(X_test), num_sequences_to_predict)
input_sequences_to_predict = X_test[random_indices]

# Predict using the LSTM model
predicted_prices = lstm_model.predict(input_sequences_to_predict)

# Denormalize the predictions manually using the minimum and maximum values of 'Close' prices
predicted_prices_denormalized = predicted_prices * (close_max - close_min) + close_min

# Plot the predicted prices alongside the actual prices from the test data
plt.figure(figsize=(12, 6))
plt.plot(actual_close_prices, label='Actual Close Prices', color='b')
plt.plot(predicted_prices_denormalized, label='LSTM Predictions', color='r')

plt.title('LSTM Model Predictions')
plt.xlabel('Time')
plt.ylabel('GBP/USD Close Price')
plt.legend()
plt.grid(True)
plt.show()