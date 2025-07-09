import tensorflow as tf
import numpy as np
from PIL import Image

# Paths and settings
MODEL_PATH = "../model/recyclable_model.tflite"
TEST_IMAGE = "sample_image.jpg"
IMG_SIZE = (150, 150)

# Load and preprocess image
img = Image.open(TEST_IMAGE).resize(IMG_SIZE).convert("RGB")
input_data = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Run inference
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Show result
print("🔍 Predicted class index:", np.argmax(output))
print("🧠 Raw output vector:", output)
