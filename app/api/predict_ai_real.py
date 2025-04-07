import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

TF_ENABLE_ONEDNN_OPTS = 0
class_labels = ['Fake', 'Real']

if len(sys.argv) < 3:
    print("Error: Image path and model name required.")
    sys.exit(1)

image_path = sys.argv[1]
model_name = sys.argv[2]

model_path = r"C:/Users/Admin/ML_flask/app/models/deep_learning/final_model_images.keras"

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

print(f"[INFO] Loading model from: {model_path}")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

try:
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]

    print(predicted_class)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)
