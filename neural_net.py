import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
from itertools import chain

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the images
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,)))

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Save the model weights and biases to separate header files
layer_weights = model.layers[0].get_weights()[0]
layer_biases = model.layers[0].get_weights()[1]

# Save layer weights
with open("layer_weight.h", 'w') as f:
    weights_str = ', '.join(map(str, layer_weights.flatten()))
    f.write("float layer_weight[" + str(784 * 10) + "]={" + weights_str + "};\n")

# Save layer biases
with open("layer_bias.h", 'w') as f:
    biases_str = ', '.join(map(str, layer_biases.flatten()))
    f.write("float layer_bias[10]={" + biases_str + "};\n")

# Predict using the trained model
test_img = Image.open("/Users/hr/my_files/test/test_4.bmp")  # Change this to the desired image path
test_img_array = np.array(test_img).astype('float32') / 255
test_img_array = test_img_array.reshape((1, 784))

predictions = model.predict(test_img_array)
print("Predictions:", predictions)

# Convert to binary predictions
binary_predictions = (predictions > 0.5).astype('int')
print("Binary Predictions:", binary_predictions)
