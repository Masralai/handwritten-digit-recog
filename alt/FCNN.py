import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

#--------------------
# preprocessing
#--------------------


mnist = tf.keras.datasets.mnist
# x->pixel data or image itself , y-> classifcation
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# we will scale down from 0-255 to 
#we will normalize the pixels and not the digits as it makes it easier for teh NNs to do ythe calcs
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#--------------------
# working on the NN
#--------------------

model = tf.keras.models.Sequential()
#flatten will convert the 28x28 grid into a line of pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) 
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
#softmax makes sure all the 10 neurons add up to 1
#softmax gives probablity for each digit to be the right answer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# sparse_categorical_crossentropy is used because the task is a multi-class classification (digits 0-9)
# adam is a optimizer that adjusts the learning rate during training.

#--------------------
# training
#--------------------

model.fit(x_train, y_train , epochs=100)

model.save('digit-recog-model.keras')


model=tf.keras.models.load_model('digit-recog-model.keras')

# loss,accuracy = model.evaluate(x_test, y_test)

# print(loss) 
# print(accuracy) 


image_number =1
while os.path.isfile(f"digits/digit_{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit_{image_number}.png")[:,:,0]
        img = np.invert(np.array([img])) #inverting as image is white on black and not vice versa
        prediction = model.predict(img)
        #np.argmax() gives us the index of the field with highest number 
        #basically the neuron with the hightest activation
        print(f"this digit is might be {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")    
    finally:
        image_number+=1    