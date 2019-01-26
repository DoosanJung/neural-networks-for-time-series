'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
from keras import backend as K
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers
import matplotlib.pyplot as plt
import argparse
from util import DecodeKerasString

class BinaryClassification(object):
    def __init_(self):
        pass

    def load_data(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data    
        self.test_labels = test_labels
        
    def preprocessing(self):
        """
        We cannot feed lists of integers into a neural network. 
        We have to turn our lists into tensors. There are two ways we could do that:

        (1) We could pad our lists so that they all have the same length, and turn them into an 
        integer tensor of shape (samples, word_indices), 
        then use as first layer in our network a layer capable of handling such integer tensors 
        (the Embedding layer).

        (2) We could one-hot-encode our lists to turn them into vectors of 0s and 1s. 
        Concretely, this would mean for instance turning the sequence [3, 5] into a 10,000-dimensional vector 
        that would be all-zeros except for indices 3 and 5, which would be ones. 
        Then we could use as first layer in our network a Dense layer, capable of 
        handling floating point vector data.
        """
        def vectorize_sequences(sequences, dimension=10000):
            # Selected (2). Create an all-zero matrix of shape (len(sequences), dimension)
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.  # set specific indices of results[i] to 1s
            return results
        
        self.x_train = vectorize_sequences(train_data)
        self.x_test = vectorize_sequences(test_data)
        self.y_train = np.asarray(self.train_labels).astype('float32')
        self.y_test = np.asarray(self.test_labels).astype('float32')

    def build_model(self):
        """
        Our input data is simply vectors, and our labels are scalars (1s and 0s): 
        this is the easiest setup you will ever encounter.

        A type of network that performs well on such a problem would be a simple stack of 
        fully-connected (Dense) layers with relu activations.

        We chose 16 as the number of "hidden units" of the layer
        """
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        # Binary classification problem, the output of our network is a probability 
        # => best to use the binary_crossentropy loss.
        # Crossentropy is usually the best choice when dealing with models that output probabilities. 
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=[metrics.binary_accuracy])
    
    def build_regularized_model(self):
        """
        other options are:
        regularizers.l1(0.001)
        regularizers.l1_l2(l1=0.001, l2=0.001). 
        
        Implementation is as follows:
            if self.l1:
                regularization += K.sum(self.l1 * K.abs(x))
            if self.l2:
                regularization += K.sum(self.l2 * K.square(x))
        https://github.com/keras-team/keras/blob/master/keras/regularizers.py
        """
        print("add L2 penalty to all layers")
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                    activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                    activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=[metrics.binary_accuracy])

    def build_dropout_model(self):
        """
        The "dropout rate" is the fraction of the features that are being zeroed-out; 
        it is usually set between 0.2 and 0.5. At test time, no units are dropped out, 
        and instead the layer's output values are scaled down by a factor equal to the dropout rate.
        """
        print("add dropout layers")
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=[metrics.binary_accuracy])

    def partially_validate_model(self, show=True):
        """
        just to show how to visualize
        """
        x_val = self.x_train[:10000]
        partial_x_train = self.x_train[10000:]
        y_val = self.y_train[:10000]
        partial_y_train = self.y_train[10000:]
        
        history = self.model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
        
        history_dict = history.history
        print("keys in a dictionary everything that happened during training.", history_dict.keys())

        def show_val_result(history):
            acc = history.history['binary_accuracy']
            val_acc = history.history['val_binary_accuracy']   
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)
            
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.clf()   # clear figure
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()
            
        if show == True:      
            show_val_result(history)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=512)

    def test_model(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def predict(self):
        return self.model.predict(self.x_test)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    regularization_group = parser.add_mutually_exclusive_group()
    regularization_group.add_argument("-l2", "--l2", help="add L2 Penalty", action="store_true")
    regularization_group.add_argument("-d", "--dropout", help="add Dropout layers", action="store_true")
    args = parser.parse_args()
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)

    # Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
    # They are split into 25,000 reviews for training and 25,000 reviews for testing, 
    # each set consisting in 50% negative and 50% positive reviews.
    # It has already been preprocessed: the reviews (sequences of words) have been turned into 
    # sequences of integers, where each integer stands for a specific word in a dictionary.
    # num_words=10000 means that we will only keep the top 10,000 most frequently words in the training data.
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print("train_data.shape: ",train_data.shape)
    print("test_data.shape: ",test_data.shape)
    print("len(train_labels): ",len(train_labels))
    print("len(test_labels): ",len(test_labels))
    print("test_data[0][:10]", test_data[0][:10])
    print("train_labels[:10]", train_labels[:10]) # 0 or 1
    print("maximum number of words in train data: ", max([max(sequence) for sequence in train_data]))
    print(DecodeKerasString.decode_keras_string(imdb, single_train_data=train_data[0]))
    
    binary_classification = BinaryClassification()
    binary_classification.load_data(train_data, train_labels, test_data, test_labels)
    binary_classification.preprocessing()
    if args.l2:
        binary_classification.build_regularized_model()
    elif args.dropout:
        binary_classification.build_dropout_model()
    else:
        binary_classification.build_model()
    binary_classification.partially_validate_model()
    binary_classification.train_model()
    test_loss, test_acc = binary_classification.test_model()
    print("test_acc: ",test_acc)

    pred = binary_classification.predict()
    print("pred_result: ", pred)