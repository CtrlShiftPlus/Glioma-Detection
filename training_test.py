import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import tensorflow as tf

# 1. TELL THE SCRIPT WHERE YOUR DATA IS
# Change this to the folder where your .h5 slices are stored
DATA_PATH = r'PATH TO YOUR DATA' # Use 'r' before the string for Windows paths

# 2. LOAD YOUR DOWNLOADED MODEL
# Make sure 'best_brats_model.h5' is in the same folder as this script
model_path = 'best_brats_model.h5'

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found! Put the file in this folder.")
else:
    # Force TensorFlow to use CPU (prevents errors if GPU drivers aren't set up)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully on CPU!")

# 3. GET A LIST OF TEST FILES
all_files = glob.glob(os.path.join(DATA_PATH, '**', '*.h5'), recursive=True)

if len(all_files) == 0:
    print(f"Error: No .h5 files found in {DATA_PATH}")
else:
    # 4. PICK A RANDOM IMAGE AND PREDICT
    random_path = random.choice(all_files)
    print(f"Testing on: {os.path.basename(random_path)}")

    with h5py.File(random_path, 'r') as f:
        # Check keys: BraTS slices usually have 'image' and 'mask'
        # Adjust if your preprocessing used different names
        image = f['image'][:] 
        mask = f['mask'][:]

    # Model expects (Batch, Height, Width, Channels) -> (1, 240, 240, 4)
    input_tensor = np.expand_dims(image, axis=0)
    prediction = model.predict(input_tensor)[0]

    # 5. VISUALIZE
    plt.figure(figsize=(15, 5))

    # Input (MRI Channel 0 - usually FLAIR)
    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title("Input MRI")
    plt.axis('off')

    # Ground Truth (Combined tumor zones)
    plt.subplot(1, 3, 2)
    plt.imshow(np.max(mask, axis=-1), cmap='jet')
    plt.title("Actual Tumor (GT)")
    plt.axis('off')

    # Prediction (Thresholded at 0.5)
    plt.subplot(1, 3, 3)
    plt.imshow(np.max(prediction > 0.5, axis=-1), cmap='jet')
    plt.title("Model Prediction")
    plt.axis('off')

    plt.show()
