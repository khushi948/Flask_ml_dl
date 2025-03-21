import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define class labels
class_labels = ['Fake', 'Real']
TF_ENABLE_ONEDNN_OPTS=0

model_path = r"C:/Users/Admin/ML_flask/app/models/deep_learning/final_model_images.keras"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file not found: {model_path}")

# Load the model
model = tf.keras.models.load_model(model_path)


# Ensure correct arguments
if len(sys.argv) < 3:
    print("Error: Image path and model name required.")
    sys.exit(1)

# Get image path and model name
image_path = sys.argv[1]
model_name = sys.argv[2]
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

# Load model
model = tf.keras.models.load_model(model_path)

# Preprocess the image
img = Image.open(image_path)
img = img.resize((224,224))  # Resize to match model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize


# Make prediction
prediction = model.predict(img_array)
predicted_class = class_labels[int(prediction[0][0] > 0.5)]


# Print the prediction (captured by subprocess)
print(predicted_class)
