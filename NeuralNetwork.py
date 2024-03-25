import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import os
from sklearn import  model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the data
X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")

# Split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.30, random_state=42)

# Normalize input data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert y_train into one-hot format
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=17))
y_train = np.array(temp)

# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):
    temp.append(to_categorical(y_test[i], num_classes=17))
y_test = np.array(temp)

# Create simple Neural Network model
model = Sequential()

# Dense layer with 10 neurons and an activation function (sigmoid)
model.add(Dense(10, activation='sigmoid'))

# Output
model.add(Dense(17, activation='softmax'))

model.build(input_shape=(None, 400))

#model.add() is here to add layers to our model
model.summary() # Here to show the number of parameters of each layers and from our model

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test,y_test))

# Save the model
model.save('NN_model.h5')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot accuracy values in the first subplot
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

# Plot loss values in the second subplot
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()


predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

# confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), predictions)
print(cm)