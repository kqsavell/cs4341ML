# Kyle Savell, Henry Wheeler-Mackta, Richard Valente

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import itertools
from PIL import Image
import PIL.ImageOps

# Symbolic Constants
K_LOW = 5
K_HIGH = 10


def neural_network():

    # Declare Model
    model = Sequential()
    model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
    model.add(Activation('relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
    model.add(Activation('softmax'))

    # Compile Model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    split_np_x = np.split(all_images, [3900, 6500])
    split_np_y = np.split(all_labels, [3900, 6500])

    train_set_x = split_np_x[0]
    test_set_x = split_np_x[1]
    train_set_y = split_np_y[0]
    test_set_y = split_np_y[1]

    num_pixels = train_set_x.shape[1] * train_set_x.shape[2]
    x_train = train_set_x.reshape(train_set_x.shape[0], num_pixels).astype('int')
    y_train = np_utils.to_categorical(train_set_y)

    num_pixels = test_set_x.shape[1] * test_set_x.shape[2]
    x_test = test_set_x.reshape(test_set_x.shape[0], num_pixels).astype('int')
    y_test = np_utils.to_categorical(test_set_y)

    # Train Model
    history = model.fit(x_train, y_train,
                        validation_split=0.20,
                        #                       validation_data = (validation_set_i, validation_set_l),
                        epochs=1500,
                        batch_size=256)

    # Report Results
    print(history.history)

    # Confusion Matrix
    ann_np = model.predict(x_test, 256, 0, None)
    print(len(ann_np))
    print(len(test_set_y))
    ann_classes = ann_np.argmax(axis=-1)
    ann_cf = confusion_matrix(test_set_y.tolist(), ann_classes.tolist())
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.figure()
    plot_confusion_matrix(ann_cf, classes=class_labels,
                          title='Confusion matrix for ANN, without normalization')
    plt.figure()
    plot_confusion_matrix(ann_cf, classes=class_labels, normalize=True,
                          title='Confusion matrix for ANN, with normalization')
    plt.show()

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.predict(x_test)


# Helper functions for feature creation
def get_pixel_features(num_array):
    """
    Gets features for a given picture pixel array
    :param num_array:
    :return: an array of the features
    """
    array_length = len(num_array)
    array_depth = len(num_array[0])
    pixel_counter = 0
    blank_pixel_counter = 0
    highest_pixel_location = 999
    lowest_pixel_location = 0
    leftist_pixel_location = 999
    rightmost_pixel_location = 0

    for j in range(array_length):
        for k in range(array_depth):
            if num_array[j][k] > 0:
                pixel_counter += 1
                if k < highest_pixel_location:
                    highest_pixel_location = k
                if k > lowest_pixel_location:
                    lowest_pixel_location = k
                if j < leftist_pixel_location:
                    leftist_pixel_location = j
                if j > rightmost_pixel_location:
                    rightmost_pixel_location = j

            if num_array[j][k] == 0:
                blank_pixel_counter += 1

    return_array = [pixel_counter / 28*28, blank_pixel_counter, highest_pixel_location, lowest_pixel_location, leftist_pixel_location, rightmost_pixel_location]
    print(return_array)
    return return_array


# Load and Flatten Data
all_labels = np.load('labels.npy')  # Load in labels, these being the "target"
all_images = np.load('images.npy')  # Load in images, these being the "data"
flat_images = []
features = []

for image in all_images:
    temp = image.ravel()
    flat_images.append(temp)
    # append to additional feature arrays
    extra_features = get_pixel_features(image)
    features.append(extra_features)

print(len(features))
# Create Training, Validation and Test Sets
training_set_i, validation_set_i, test_set_i = [], [], []  # Image sets
training_set_f, validation_set_f, test_set_f = [], [], []  # custom feature sets
training_set_l, validation_set_l, test_set_l = [], [], []  # Label sets
i = 0

while i < 3900:
    training_set_i.append(flat_images[i])
    training_set_f.append(features[i])
    training_set_l.append(all_labels[i])
    i += 1
while i < 4875:
    validation_set_i.append(flat_images[i])
    validation_set_f.append(features[i])
    validation_set_l.append(all_labels[i])
    i += 1
while i < 6500:
    test_set_i.append(flat_images[i])
    test_set_f.append(features[i])
    test_set_l.append(all_labels[i])
    i += 1

predicted_labels = []  # Set of predicted labels generated by the algorithms


# Class for the Decision Tree Classification
class DecisionTreeClassification:
    def __init__(self, data, target, tree):
        """
        DecisionTreeClassification takes an array of data
        :param data: the training data
        :param target: the training target
        :param: tree: the DecisionTreeClassifier
        """
        self.data = data
        self.target = target
        self.estimator = tree

    def fit(self):
        """
        Builds the decision tree using scikit's decision tree classifiers
        :return: the classified tree
        """
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

    def predict(self, x_test):
        """
        Predicts based on some test input and prints how it stacks up against truth
        Please calla after "classify"
        :param x_test: the test array of inputs (in our case, the pictures)
        :return: the result
        """
        predict = self.estimator.predict(x_test)
        return predict


# Function from scikit-learn documentation:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


# K-Nearest Neighbors
def k_nearest():
    cur_k = K_LOW
    best_k = 0
    best_sum = 0

    # Find best k value using validation set
    while cur_k <= K_HIGH:
        neigh = KNeighborsClassifier(n_neighbors=cur_k)
        neigh.fit(training_set_i, training_set_l)
        predicted_data = neigh.predict(validation_set_i)
        cf = confusion_matrix(validation_set_l, predicted_data)
        print("k = "+str(cur_k))
        print(cf)
        print("\n")
        cur_val = 0
        sum_correct = 0
        while cur_val < 10:
            sum_correct += cf[cur_val][cur_val]
            cur_val += 1
        if sum_correct > best_sum:
            best_sum = sum_correct
            best_k = cur_k
        cur_k += 1

    # Use best k on test set
    print("Best k-value is "+str(best_k))
    neigh = KNeighborsClassifier(n_neighbors=cur_k)
    neigh.fit(training_set_i, training_set_l)
    predicted_data = neigh.predict(test_set_i)
    cf = confusion_matrix(test_set_l, predicted_data)
    print(cf)
    return cf


# Run the program
def main():
    print("Starting...")

    # Neural Network
    print("Artificial Neural Network on Full Pixel Data: ")
    neural_network()

    # Decision Tree
    print("Decision Tree Working on Full Pixel Data: ")
    dt = DecisionTreeClassifier(max_depth=19, min_samples_leaf=1, splitter="best", random_state=None, criterion="entropy")
    dt = DecisionTreeClassification(training_set_i, training_set_l, dt)
    dt.fit()

    print('Accuracy on the training subset: {:.3f}'.format(dt.score_accuracy(training_set_i, training_set_l)))
    print('Accuracy on the validation subset: {:.3f}'.format(dt.score_accuracy(validation_set_i, validation_set_l)))
    print('Accuracy on the test subset: {:.3f}'.format(dt.score_accuracy(test_set_i, test_set_l)))
    print('Confusion matrix on validation subset:')
    cf1 = confusion_matrix(validation_set_l, dt.predict(validation_set_i))

    print("Decision Tree Working on Hand-Engineered Features: ")
    dt2 = DecisionTreeClassifier(max_depth=19, random_state=0, min_samples_leaf=1, )
    dt2 = DecisionTreeClassification(training_set_f, training_set_l, dt2)
    dt2.fit()

    print('Accuracy on the training hand-engineered feature subset: {:.3f}'.format(dt2.score_accuracy(training_set_f,
                                                                                                      training_set_l)))
    print('Accuracy on the validation hand feature subset: {:.3f}'.format(dt2.score_accuracy(validation_set_f,
                                                                                             validation_set_l)))
    print('Accuracy on the test hand-engineered feature subset: {:.3f}'.format(dt2.score_accuracy(test_set_f,
                                                                                                  test_set_l)))
    print('Confusion matrix on test hand-engineered feature subset:')
    cf2 = confusion_matrix(validation_set_l, dt2.predict(validation_set_f))

    # K-Nearest Neighbors
    print("K-nearest on all pixel-data")
    cf = k_nearest()
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Plot non-normalized confusion matrix for k-nearest
    plt.figure()
    plot_confusion_matrix(cf, classes=class_labels,
                          title='Confusion matrix for K-Nearest, without normalization')

    # Plot normalized confusion matrix for k-nearest
    plt.figure()
    plot_confusion_matrix(cf, classes=class_labels, normalize=True,
                          title='Normalized confusion matrix for K-Nearest')

    # plot non-normalized confusion matrix for validation set for DT
    plt.figure()
    plot_confusion_matrix(cf1, classes=class_labels,
                          title='Confusion matrix for variation DT, without normalization')

    # plot normalized confusion matrix for validation set for DT
    plt.figure()
    plot_confusion_matrix(cf1, classes=class_labels, normalize=True,
                          title='Confusion matrix for variation DT, with normalization')

    plt.show()

# Sample code for converting array to image
# img1 = Image.fromarray(all_images[4875 + 70], "L")
# img1_inv = PIL.ImageOps.invert(img1)
# img1_inv.show()

main()
