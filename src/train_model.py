import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "../dataset"  # Update if dataset is elsewhere
MODEL_SAVE_PATH = "../model/recyclable_model.h5"
