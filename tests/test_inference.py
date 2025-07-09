import numpy as np
from demo.run_inference import preprocess_image

# Test: Image shape after preprocessing
def test_image_shape():
    img_array = preprocess_image("demo/sample_image.jpg")
    assert img_array.shape == (1, 150, 150, 3)

# Test: Output dimension check (if model has 3 classes)
def test_output_size():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="../model/recyclable_model.tflite")
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    assert output_details[0]['shape'][1] == 3  # Replace with actual class count
