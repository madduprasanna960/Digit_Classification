import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the Model
        model = Sequential()

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(height, width, depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Third CONV => RELU => POOL Layer
        model.add(Conv2D(100, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # If the saved_weights file is already present, i.e., model is pre-trained, load the weights
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model

# Load MNIST data
print('\nLoading MNIST Data...')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
print('\nPreprocessing Data...')
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

# Convert class vectors to binary class matrices
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Create CNN model
print('\nCreating CNN Model...')
cnn_model = CNN.build(width=28, height=28, depth=1, total_classes=10)

# Compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
print('\nTraining CNN Model...')
cnn_model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=200, verbose=2)

# Evaluate the model
print('\nEvaluating CNN Model...')
evaluation_scores = cnn_model.evaluate(test_images, test_labels, verbose=0)
print("CNN Test Accuracy: {:.2f}%".format(evaluation_scores[1] * 100))

# Save the model weights with the correct filename extension
weights_filename = 'cnn_mnist_weights.weights.h5'
cnn_model.save_weights(weights_filename)
print(f'Model weights saved to {weights_filename}')

# Plot some predictions
predictions = cnn_model.predict(test_images)

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {np.argmax(test_labels[i])}, Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()
