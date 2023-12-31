from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# preprocessing the training set

train_datagen = ImageDataGenerator(rescale=1./255,     # processing the dataset to prevent overfitting
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set',  # importing the dataset
                                                 # final size of image
                                                 target_size=(64, 64),
                                                 batch_size=32,          # no of images in each batch
                                                 class_mode='binary')    # binary or categorical


# preprocessing the test set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

# initializing the Cnn

cnn = tf.keras.models.Sequential()

# convolution first layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))

# pooling first layer

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# adding a second convolution layer

# no need for input shape for second layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# flattening  1 dimensional

cnn.add(tf.keras.layers.Flatten())

# Full connection

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer

# binary results obtained
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the cnn

cnn.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Making a single prediction


test_image = image.load_img('test_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
print(prediction)
