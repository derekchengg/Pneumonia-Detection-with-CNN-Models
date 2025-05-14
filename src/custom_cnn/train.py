import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to supress TensorFlow warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "..", "processed_data")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    print("Loaded X_train.npy, shape:", X_train.shape)

    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    print("Loaded X_val.npy, shape:", X_val.shape)

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    print("Loaded X_test.npy, shape:", X_test.shape)

    
    # classify data for data augmentation
    X_pneumonia = X_train[y_train == 0]
    y_pneumonia = y_train[y_train == 0]
    X_normal = X_train[y_train == 1]
    y_normal = y_train[y_train == 1]
    
    pneumonia_count = len(X_pneumonia)
    normal_count = len(X_normal)

    # data augmentation
    data_aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        shear_range = 0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.1,
    )

    # Oversample NORMAL class to match PNEUMONIA count
    X_augmented = []
    y_augmented = []
    needed = pneumonia_count - normal_count

    for x_batch, y_batch in data_aug.flow(X_normal, y_normal, batch_size=32):
        for x, y in zip(x_batch, y_batch):
            X_augmented.append(x)
            y_augmented.append(y)
            if len(X_augmented) >= needed:
                break
        if len(X_augmented) >= needed:
            break

    X_balanced = np.concatenate((X_train, np.array(X_augmented)))
    y_balanced = np.concatenate((y_train, np.array(y_augmented)))
    
    
    # Check the class distribution to see if the augmentation balanced the dataset
    unique, counts = np.unique(y_balanced, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("\n\nClass distribution in X_balanced:", class_distribution,"\n\n")


    img_size = 175
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Dropout(0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    model.summary()    
    
    idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[idx]
    y_balanced = y_balanced[idx]
        
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, "..", "..", "results", "custom_cnn_model.h5"),
        monitor='val_accuracy',             
        save_best_only=True,            
        save_weights_only=False,        
        verbose=1                       
    )
    
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=1
    )
    
    history = model.fit(
        X_balanced, y_balanced,
        batch_size=32,
        validation_data=(X_val, y_val),
        epochs=18,
        verbose=1,
        callbacks=[checkpoint, lr_schedule, early_stopping]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%","\n\n")

    y_pred_probs = model.predict(X_val)

    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = y_val

    print(classification_report(y_true, y_pred, target_names=["PNEUMONIA", "NORMAL"]))


    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "..", "..", "results", "training_curves.png"))
    
    
if __name__ == "__main__":
    main()