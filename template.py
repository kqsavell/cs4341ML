# Kyle Savell, Henry Wheeler-Mackta, Richard Valente

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

# Load and Flatten Data
all_labels = np.load('labels.npy')  # Load in labels, these being the "target"
all_images = np.load('images.npy')  # Load in images, these being the "data"
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
#
# # Model Template
# model = Sequential() # declare model
# model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
# model.add(Activation('relu'))
# #
# #
# #
# # Fill in Model Here
# #
# #
# model.add(Dense(10, kernel_initializer='he_normal')) # last layer
# model.add(Activation('softmax'))
#
#
# # Compile Model
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train Model
# history = model.fit(x_train, y_train,
#                     validation_data = (x_val, y_val),
#                     epochs=10,
#                     batch_size=512)
#
#
# # Report Results
#
# print(history.history)
# model.predict()

# The above stuff should prolly be put into classes


# Class for the Decision Tree Classification
class DecisionTreeClassification:
    def __init__(self, data, target):
        """
        DecisionTreeClassification takes an array of data
        :param data: the training data
        :param target: the training target
        """
        self.data = data
        self.target = target
        self.estimator = DecisionTreeClassifier(self.data, self.target)

    def classify(self):
        """
        Builds the decision tree using scikit's decision tree classifiers
        :return: the classified tree
        """
        self.estimator = DecisionTreeClassifier(max_lead_nodes=3, random_state=0)
        self.estimator.fit(self.data, self.target)
        return self.estimator

    def score_accuracy(self, x_input, y_input):
        """
        Returns the accuracy score for the tree given some images and labels
        :param x_input: array of inputs (in our case, the pictures)
        :param y_input: truth array (in our case, the labels)
        :return: the accuracy score
        """
        return self.estimator.score(x_input, y_input)

    def predict(self, x_test, y_test):
        """
        Predicts based on some test input and prints how it stacks up against truth
        Please calla after "classify"
        :param x_test: the test array of inputs (in our case, the pictures)
        :param y_test: the truth array (in our case, the labels)
        """
        predict = self.estimator.predict(x_test)
        accuracy_score(y_test, predict)


def confusion_matrix(y_true, y_predicted):
    """
    Returns a confusion matrix (two-dimensional array)
    :param y_true: array of what should've happened
    :param y_predicted: array of what was predicted
    :return: the array
    """
    return confusion_matrix(y_true, y_predicted)


# run the program
def main():
    print("Starting...")
    dt = DecisionTreeClassification(training_set_i, training_set_l)



main()