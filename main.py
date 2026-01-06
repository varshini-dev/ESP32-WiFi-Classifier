import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("wifi_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save normalization values
np.save("mean.npy", scaler.mean_)
np.save("std.npy", scaler.scale_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model (TinyML-friendly)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("wifi_model.tflite", "wb") as f:
    f.write(tflite_model)

with open("wifi_model.tflite", "rb") as f:
    data = f.read()

with open("wifi_model.h", "w") as f:
    f.write("unsigned char wifi_model_tflite[] = {")
    f.write(",".join(hex(b) for b in data))
    f.write("};\n")
    f.write(f"unsigned int wifi_model_tflite_len = {len(data)};")

