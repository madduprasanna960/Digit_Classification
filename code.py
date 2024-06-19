import sys
import os

# Add the MNIST_Dataset_Loader directory to the system path
sys.path.append(os.path.abspath(r'C:\Users\prasanna\Desktop\Handwritten-Digit-Recognition-using-Deep-Learning-main\Handwritten-Digit-Recognition-using-Deep-Learning-main\MNIST_Dataset_Loader'))

from mnist_loader import MNIST
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

old_stdout = sys.stdout
log_file = open("summary.log", "w")
sys.stdout = log_file

print('\nLoading MNIST Data...')
data = MNIST(os.path.abspath(r'C:\Users\prasanna\Desktop\Handwritten-Digit-Recognition-using-Deep-Learning-main\Handwritten-Digit-Recognition-using-Deep-Learning-main\MNIST_Dataset_Loader\dataset'))

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

# Features
X = train_img

# Labels
y = train_labels

print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

print('\nRandom Forest Classifier with n_estimators = 100, n_jobs = 10')
print('\nPickling the Classifier for Future Use...')
clf = RandomForestClassifier(n_estimators=100, n_jobs=10)
clf.fit(X_train, y_train)

with open('MNIST_RFC.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('MNIST_RFC.pickle', 'rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
confidence = clf.score(X_test, y_test)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test, y_pred)

print('\nRFC Trained Classifier Confidence: ', confidence)
print('\nPredicted Values: ', y_pred)
print('\nAccuracy of Classifier on Validation Image Data: ', accuracy)
print('\nConfusion Matrix: \n', conf_mat)

# Plot Confusion Matrix Data as a Matrix
plt.matshow
