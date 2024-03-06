# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# Sample values for X, Y, Z
sample_x = 0.5
sample_y = -0.2
sample_z = 0.8

# Process X, Y, Z values as needed
print(f"Received X: {sample_x}, Y: {sample_y}, Z: {sample_z}")

# Load Dataset
dataset_path = 'my_model.h5'  # Replace with the path to your dataset
activity_folders = os.listdir(dataset_path)
print(activity_folders)

# ... (Rest of your existing code for loading and preprocessing data)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=400, callbacks=[checkpoint], verbose=1)

# Evaluate the model
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=400, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
max_y_pred_train = np.argmax(y_pred_train, axis=1)

LABELS = ['Going_Up_Down_Stairs', 'Standing', 'Talking_while_Standing', 'Walking',
          'Walking_and_Talking_with_Someone', 'Working_at_Computer']

y_pred_test = model.predict(X_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(matrix, cmap='PiYG_r', linecolor='white', linewidths=1,
            xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
