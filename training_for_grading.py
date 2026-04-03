#1
from google.colab import files
files.upload()

#2

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#3

!kaggle datasets download -d awsaf49/brats2020-training-data

#4

!unzip -q brats2020-training-data.zip -d /content/data
print("Done")


#5

import os

# 1. Check folder contents
files_in_dir = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
print(f"Total .h5 files found in {DATA_DIR}: {len(files_in_dir)}")
if len(files_in_dir) > 0:
    print(f"Sample filename from folder: '{files_in_dir[0]}'")

# 2. Check CSV contents
print(f"Total IDs in mapping_df: {len(mapping_df)}")
if len(mapping_df) > 0:
    print(f"Sample ID from CSV: '{mapping_df['BraTS_2020_subject_ID'].iloc[0]}'")

# 3. Check the intersection
available_ids = [f.replace('.h5', '') for f in files_in_dir]
matches = mapping_df[mapping_df['BraTS_2020_subject_ID'].isin(available_ids)]
print(f"Number of matches found: {len(matches)}")


#6

import re

# --- 1. Create a Numeric Label Map ---
# Convert 'BraTS20_Training_001' -> 1
def extract_id_from_csv(text):
    match = re.search(r'(\d+)$', text)
    return int(match.group(1)) if match else None

mapping_df['num_id'] = mapping_df['BraTS_2020_subject_ID'].apply(extract_id_from_csv)
# Create dict: { 1: 1 (HGG), 2: 0 (LGG) ... }
id_to_label = dict(zip(mapping_df['num_id'], mapping_df['label']))

# --- 2. Filter and Match Files ---
all_h5_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
final_files = []
final_labels = []

print("Starting file matching...")
for f in all_h5_files:
    # Extract '94' from 'volume_94_slice_117.h5'
    match = re.search(r'volume_(\d+)_', f)
    if match:
        vol_num = int(match.group(1))
        if vol_num in id_to_label:
            final_files.append(f)
            final_labels.append(id_to_label[vol_num])

patient_list = np.array(final_files)
labels_list = np.array(final_labels)

print(f"✅ Successfully matched {len(patient_list)} slices to their volume labels.")

# --- 3. Update Generator to load exact filenames ---
class BratsH5GradingGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames, labels, batch_size=32): # Increased batch size for 57k slices
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        X = []
        for f_name in batch_files:
            file_path = os.path.join(DATA_DIR, f_name)
            with h5py.File(file_path, 'r') as f:
                img = f['image'][:] 
                X.append(img.astype(np.float32))
                
        return np.array(X), np.array(batch_labels)

# --- 4. Split and Train ---
X_train, X_val, y_train, y_val = train_test_split(
    patient_list, labels_list, test_size=0.15, random_state=42
)

train_gen = BratsH5GradingGenerator(X_train, y_train)
val_gen = BratsH5GradingGenerator(X_val, y_val)

model = build_grading_model()
# With 57,000 slices, 3-5 epochs is usually enough to converge
model.fit(train_gen, validation_data=val_gen, epochs=5)

model.save("brats_grading_model.h5")
files.download("brats_grading_model.h5")



"""Gemini said
Yes, this approach incorporates several Big Data and Efficient Deep Learning methodologies to handle the scale of the BraTS dataset (57,000+ files) without crashing your system.

Here is a breakdown of the specific big data techniques being used in your Colab script:

1. Data Streaming (The "Generator" Pattern)
The most critical methodology used is the Data Generator (tf.keras.utils.Sequence).

The Problem: 57,000 .h5 files would take up roughly 40GB–80GB of RAM if loaded all at once. Google Colab only provides ~12GB–16GB.

The Solution: Instead of loading the entire dataset, the generator uses Lazy Loading. it only pulls 32 images (one "batch") into RAM at a exact moment they are needed for a training step, then flushes them to make room for the next 32.

2. Disk-to-Tensor Pipeline
By keeping the data as individual .h5 files on the disk and reading them via h5py during training, we are using Disk-Resident Data Processing.

This is a standard "Big Data" approach where the bottleneck shifts from Memory (RAM) to I/O (Disk Speed).

H5 is specifically a "Big Data" format because it supports fast, parallel I/O and hierarchical storage, making it much faster than reading 57,000 individual PNG or JPEG files.

3. Dimensionality Reduction (3D to 2D)
The original BraTS data is 3D (volumes). Processing 3D volumes is exponentially more "expensive" in terms of computation.

By using 2D Slices, we reduce the data complexity while keeping the most relevant features.

Because you have 57,000 slices, you are essentially performing Data Augmentation via Sampling. Every slice of a single patient’s brain provides a slightly different "view" of the tumor, which helps the model generalize better than if it only saw one 3D volume per patient.

4. Vectorized Mapping & Regex Pre-processing
In the latest code, we used Vectorized Dictionary Mapping and Regular Expressions (Regex) to link the files to the CSV.

Instead of nested loops (which would be O(N 
2
 ) and take forever with 57k files), we created a Hash Map (Dictionary) for the labels.

This turns the search into an O(1) operation, allowing us to match all 57,000 files to their labels in a fraction of a second.

5. Mini-Batch Stochastic Gradient Descent (SGD)
By setting batch_size=32, we are using Mini-Batching. This is a core Big Data optimization. It allows the model to update its "knowledge" (weights) based on a small representative sample of the data rather than waiting to see all 57,000 images, which would be computationally impossible in a single pass.

Is there a "Bigger" way?
If you had millions of images, the next step would be Distributed Training (using tf.distribute.MirroredStrategy to use multiple GPUs) or converting the data into TFRecords (Google's binary storage format optimized for TensorFlow). However, for 57,000 slices on a single Colab GPU, the H5 Generator is the "sweet spot" for efficiency and speed.

Are you ready to start the training run, or should we adjust the batch size to speed it up further?"""