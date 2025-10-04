# Task8_traffic_sign_recognition.py
# Requirements: tensorflow, keras, numpy, pandas, sklearn, opencv-python (cv2)
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = "gtsrb"  # expected structure: DATA_DIR/train/{class_id}/*.ppm or png
IMG_SIZE = (64,64)
batch_size = 64
num_classes = len(os.listdir(os.path.join(DATA_DIR,'train')))

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.1, zoom_range=0.1, horizontal_flip=False)

train_gen = train_datagen.flow_from_directory(os.path.join(DATA_DIR,'train'),
                                              target_size=IMG_SIZE, batch_size=batch_size, subset='training')
val_gen = train_datagen.flow_from_directory(os.path.join(DATA_DIR,'train'),
                                            target_size=IMG_SIZE, batch_size=batch_size, subset='validation')

# Simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0],IMG_SIZE[1],3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=20)

model.save('traffic_sign_cnn.h5')

# For transfer learning: use MobileNetV2 base with global pooling and new head (fine-tune last blocks)
