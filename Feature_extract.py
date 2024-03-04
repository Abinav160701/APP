import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from models import load_model_from_db
# Feature extraction function
def extract_features(l1,l2):
    model = load_model_from_db('Men','Clothing')
    directory=f'Database/DB_{l1}-{l2}'
    features = []
    labels = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                img_features = model.predict(img_array)
                features.append(img_features.flatten())
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return features, labels