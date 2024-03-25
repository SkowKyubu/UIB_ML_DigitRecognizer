import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the data
X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")

X = X[:25000]
y = y[:25000]
# Split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.30, random_state=42)

# Defining the parameters grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.1, 1],
              'kernel': ['rbf', 'poly']}

# Creating a support vector classifier
svc = svm.SVC(probability=True)

# Creating a model using GridSearchCV with the parameters grid
model = GridSearchCV(svc, param_grid)

model.fit(X_train,y_train)

# Testing the model using the testing data
y_pred = model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"The model is {accuracy * 100}% accurate")

predictions = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)