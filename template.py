# Kyle Savell, Henry Wheeler-Mackta, Richard Valente

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Load and Flatten Data
all_labels = np.load('labels.npy')  # Load in labels
all_images = np.load('images.npy')  # Load in images
flat_images = []

for image in all_images:
    temp = image.ravel()
    flat_images.append(temp)

    # Create Training, Validation and Test Sets
    training_set_i, validation_set_i, test_set_i = [], [], []  # Image sets
    training_set_l, validation_set_l, test_set_l = [], [], []  # Label sets
    i = 0
    while i < 3900:
        training_set_i.append(flat_images[i])
        training_set_l.append(all_labels[i])
        i += 1
    while i < 4875:
        validation_set_i.append(flat_images[i])
        validation_set_l.append(all_labels[i])
        i += 1
    while i < 6500:
        test_set_i.append(flat_images[i])
        test_set_l.append(all_labels[i])
        i += 1

# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=10, 
                    batch_size=512)


# Report Results

print(history.history)
model.predict()
