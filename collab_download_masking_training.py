from google.colab import files
import os

# 1. Upload your kaggle.json
files.upload()

# 2. Setup Kaggle directory
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 3. Download the dataset directly (Using the URL suffix from Kaggle)
!kaggle datasets download -d awsaf49/brats2020-training-data

# 4. Unzip the data
!unzip -q brats2020-training-data.zip -d ./data
print("Data downloaded and unzipped successfully!")


#5

import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Update this path to where you unzipped the data in Colab
DATA_PATH = '/content/data/' 

# Find all .h5 files
all_h5_files = sorted(glob.glob(DATA_PATH + '**/*.h5', recursive=True))
print(f"Total slices found: {len(all_h5_files)}")

# Split into Train and Validation (80/20)
train_files, val_files = train_test_split(all_h5_files, test_size=0.2, random_state=42)


#6

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- DATA GENERATOR ---
class BraTS2DGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size=32):
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, index):
        batch = self.file_list[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        for f_path in batch:
            with h5py.File(f_path, 'r') as f:
                # Replace 'image' and 'mask' with the actual dataset names inside your HDF5 files
                X.append(f['image'][:])
                y.append(f['mask'][:])
        return np.array(X), np.array(y)

# --- SIMPLE 2D U-NET ---
def build_unet(input_shape=(240, 240, 4)):
    inputs = layers.Input(input_shape)
    # Down
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)
    # Bottom
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    # Up
    u1 = layers.UpSampling2D()(b)
    concat = layers.Concatenate()([u1, c1])
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat)
    # Output (3 channels for the 3 tumor zones)
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c2)
    
    return models.Model(inputs, outputs)

# Instantiate the model
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#7


train_gen = BraTS2DGenerator(train_files, batch_size=16)
val_gen = BraTS2DGenerator(val_files, batch_size=16)

# Save the model to Colab's local storage
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_brats_model.h5', save_best_only=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10, 
    callbacks=[checkpoint]
)