import sys
import os
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

# Load training data
print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

# Load testing data
print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

# Prepare training and validation data
X_train, X_val, y_train, y_val = model_selection.train_test_split(train_img, train_labels, test_size=0.1)

# Initialize classifiers
clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=10)
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_svm = SVC()

# Train classifiers
print('\nTraining Random Forest Classifier...')
clf_rf.fit(X_train, y_train)
print('\nTraining K-Nearest Neighbors Classifier...')
clf_knn.fit(X_train, y_train)
print('\nTraining Support Vector Machine Classifier...')
clf_svm.fit(X_train, y_train)

# Validate classifiers
print('\nValidating Classifiers...')
val_pred_rf = clf_rf.predict(X_val)
val_pred_knn = clf_knn.predict(X_val)
val_pred_svm = clf_svm.predict(X_val)

# Calculate accuracy
acc_rf = accuracy_score(y_val, val_pred_rf)
acc_knn = accuracy_score(y_val, val_pred_knn)
acc_svm = accuracy_score(y_val, val_pred_svm)

# Test classifiers
print('\nTesting Classifiers...')
test_pred_rf = clf_rf.predict(test_img)
test_pred_knn = clf_knn.predict(test_img)
test_pred_svm = clf_svm.predict(test_img)

# Calculate test accuracy
test_acc_rf = accuracy_score(labels_test, test_pred_rf)
test_acc_knn = accuracy_score(labels_test, test_pred_knn)
test_acc_svm = accuracy_score(labels_test, test_pred_svm)

# Display results
print('\nRandom Forest Validation Accuracy: ', acc_rf)
print('K-Nearest Neighbors Validation Accuracy: ', acc_knn)
print('Support Vector Machine Validation Accuracy: ', acc_svm)

print('\nRandom Forest Test Accuracy: ', test_acc_rf)
print('K-Nearest Neighbors Test Accuracy: ', test_acc_knn)
print('Support Vector Machine Test Accuracy: ', test_acc_svm)

# Compare predictions for random images
a = np.random.randint(0, len(test_img), 4)

for i in a:
    two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
    plt.title(f'Original Label: {test_labels[i]}')
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()
    print(f'Random Forest Prediction: {test_pred_rf[i]}')
    print(f'KNN Prediction: {test_pred_knn[i]}')
    print(f'SVM Prediction: {test_pred_svm[i]}')

# Plot comparison graph
models = ['Random Forest', 'KNN', 'SVM']
val_accuracies = [acc_rf, acc_knn, acc_svm]
test_accuracies = [test_acc_rf, test_acc_knn, test_acc_svm]

x = np.arange(len(models))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, val_accuracies, width, label='Validation')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by classifier and dataset')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()

plt.show()
