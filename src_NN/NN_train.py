"""
Training Environment:
Python version = 3.7.X
Tensorflow-GPU = 2.6.0
Keras = 2.6.0 (=Tensorflow-GPU)
CUDA version = 12.1
cuDNN version = 8.8.1.3(cuda12)

Building the simple neural network(NN) with Flatten,and Full Connection layers.
Dataset uses CSI amplitudes directly.
Results and models will export to Simple_NN directory.
"""
import csv
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Flatten, Dense
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

# Define parameters
epoch_size = 300
batch_size = 32

# Import Dataset
dataset = pd.read_csv('csi_amplitudes.csv')
X = dataset.iloc[:, 1:53].values
y = dataset.iloc[:, 53].values

# Transform to NumPy array
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# Initialize the Neural Network(NN)
NN_model = Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# # Simple NN
# NN_model.add(Flatten())
# NN_model.add(Dense(units=128, activation='relu'))
# NN_model.add(Dense(units=1, activation='sigmoid'))

# Define Learning Rate
initial_learning_rate = 0.001
decay_steps = epoch_size//4
decay_rate = 0.96
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Compiling the NN
optimizer = Adam(learning_rate=lr_schedule)
NN_model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

csv_logger = CSVLogger('result/train/csv/train_epoch300_logs.csv', append=False)

history = NN_model.fit(X_train,
                       y_train,
                       epochs=epoch_size,
                       batch_size=batch_size,
                       steps_per_epoch=246,  # MAX <= 10519*0.75//32
                       validation_data=(X_val, y_val),
                       validation_steps=16,  # MAX <= 10519*0.05//32=16
                       callbacks=[csv_logger])


NN_model.summary()
NN_model.save('model/train_model_epoch300')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('result/train/Loss_epoch300.png')
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('result/train/acc_epoch300.png')
plt.show()

# 評估模型
loss, accuracy = NN_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')