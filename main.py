import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

#--------------------
# Preprocessing
#--------------------

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalization
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomFlip("horizontal")
])

x_train = data_augmentation(x_train)

#--------------------
# CNN Model
#--------------------


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(), #converting 2d output into 1d array to feed the dense layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#--------------------
# Training
#--------------------

#early stopping when the validation accuracy stops improving to avoid overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))


model.save('optimized-cnn-model.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

#--------------------
# Preprocessing Function for images i made on paint :p
#--------------------

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  #Converts the image to grayscale (single channel)
    img = cv2.resize(img, (28, 28))  #Resizes the image to 28x28 pixels (the standard MNIST image size)
    img = np.invert(img)  #Inverts the colors (so digits appear as white on black, like MNIST)
    img = img.astype('float32') / 255.0  #Scales the pixel values to [0, 1]
    
    
    if np.sum(img) < 100:    #If the image is too thin (low pixel sum), it adds padding to ensure the digit is centered.
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (28, 28))
    
    img = img.reshape(1, 28, 28, 1)  #Reshapes the image to fit the model’s input shape.
    return img

#--------------------
# Testing 
#--------------------

model = tf.keras.models.load_model('optimized-cnn-model.keras')

image_number = 1
while os.path.isfile(f"digits/digit_{image_number}.png"):
    try:
        img = preprocess_image(f"digits/digit_{image_number}.png")
        prediction = model.predict(img)
        print(f"This digit might be {np.argmax(prediction)}")

        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
