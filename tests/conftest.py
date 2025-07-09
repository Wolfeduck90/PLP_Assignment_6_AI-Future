import pytest
import numpy as np
from PIL import Image

# Sample fixture for reusable preprocessed image
@pytest.fixture
def sample_input():
    def preprocess(img_path):
        img = Image.open(img_path).resize((150, 150)).convert('RGB')
        arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
        return arr
    return preprocess("demo/sample_image.jpg")
