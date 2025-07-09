import tensorflow as tf

# Paths
INPUT_MODEL = "../model/recyclable_model.h5"
OUTPUT_TFLITE = "../model/recyclable_model.tflite"

# Load and convert
model = tf.keras.models.load_model(INPUT_MODEL)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved: {OUTPUT_TFLITE}")
