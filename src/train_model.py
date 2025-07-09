import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
DATA_PATH = "../dataset/"
SAVE_PATH = "../model/recyclable_model.h5"

# Data loading and preprocessing
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = data_gen.flow_from_directory(DATA_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                     class_mode='categorical', subset='training')

val = data_gen.flow_from_directory(DATA_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                   class_mode='categorical', subset='validation')

# Simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and save
model.fit(train, validation_data=val, epochs=EPOCHS)
model.save(SAVE_PATH)
print(f"✅ Model saved: {SAVE_PATH}")
