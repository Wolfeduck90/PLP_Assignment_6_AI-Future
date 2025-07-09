# PLP_Assignment_6_AI-Future

README.md
markdown
# Edge AI Recyclable Item Classifier ♻️🧠

This project uses TensorFlow and TensorFlow Lite to train and deploy a lightweight image classifier that recognizes recyclable materials such as plastic bottles, cans, and cardboard. Designed for edge devices like Raspberry Pi, the model prioritizes low latency, privacy, and offline capability.

## 🔧 Folder Structure

PLP_Assignment_6_AI_Future
│
├── dataset/                         # Class-wise image folders for training/validation
│   ├── plastic_bottle/             # Example class folder
│   ├── cardboard/                  
│   └── can/                        
│
├── src/                             # Core scripts
│   ├── train_model.py               # Builds and trains the CNN model
│   ├── convert_tflite.py            # Converts model to .tflite format
│
├── model/                           # Saved models
│   ├── recyclable_model.h5          # Trained Keras model
│   └── recyclable_model.tflite      # TFLite version for edge inference
│
├── demo/                            # Deployment demo scripts
│   ├── run_inference.py             # Runs inference on sample image
│   └── sample_image.jpg             # Placeholder image for testing
│
├── tests/                           # Unit tests
│   ├── test_inference.py            # Tests for preprocessing & model output
│   └── conftest.py (optional)       # Fixtures/shared setup if needed
│
├── requirements.txt                 # Dependencies (TensorFlow, Pillow, Pytest)
└── README.md                        # Documentation with project overview and setup

## ⚙️ Key Features

- Trains a lightweight convolutional neural network (CNN)
- Converts model to `.tflite` format for edge deployment
- Runs image classification locally for recyclable detection
- Compatible with Raspberry Pi or Colab simulation

## 🚀 Getting Started

1. Prepare image dataset under `dataset/` with class-wise folders.
2. Run `src/train_model.py` to train the model.
3. Run `src/convert_tflite.py` to generate `.tflite` version.
4. Test using `demo/run_inference.py`
