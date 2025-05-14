import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to supress TensorFlow warnings

import numpy as np
import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix

def main(model_filename):
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.join(BASE_DIR, "..", "..")
    
  model_path = os.path.join(project_root, "results", model_filename)
  data_dir = os.path.join(project_root, "processed_data")

  print("Loading test data...")
  X_test = np.load(os.path.join(data_dir, "X_test.npy"))
  y_test = np.load(os.path.join(data_dir, "y_test.npy"))

  print("Loading model...")
  model = tf.keras.models.load_model(model_path)

  print("Evaluating model on test set...")
  loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
  print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

  y_pred_probs = model.predict(X_test)
  y_pred = (y_pred_probs > 0.5).astype(int).flatten()

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred, target_names=["PNEUMONIA", "NORMAL"]))

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate a saved CNN model on test data.")
  parser.add_argument("model_filename")
  args = parser.parse_args()

  main(args.model_filename)