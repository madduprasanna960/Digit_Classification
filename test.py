import sys
import os
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Add the MNIST_Dataset_Loader directory to the system path
sys.path.append(os.path.abspath('./MNIST_Dataset_Loader'))

# Load the MNIST dataset
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')

# Load testing data
print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

# Load the trained classifier from the pickle file
with open('MNIST_RFC.pickle', 'rb') as f:
    clf = pickle.load(f)

# Make predictions on test input images
print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

# Calculate accuracy of the trained classifier on test data
print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
acc = accuracy_score(test_labels, test_labels_pred)

# Create confusion matrix for test data
print('\nCreating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels, test_labels_pred)

print('\nPredicted Labels for Test Images: ', test_labels_pred)
print('\nAccuracy of Classifier on Test Images: ', acc)
print('\nConfusion Matrix for Test Data: \n', conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

# Show the Test Images with Original and Predicted Labels
a = np.random.randint(1, 30, 10)
for i in a:
    two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
    plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i], test_labels_pred[i]))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()
