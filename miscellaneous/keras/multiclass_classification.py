'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
from keras import backend as K
from keras.datasets import reuters
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from util import DecodeKerasString

class MulticlassClassification(object):
    """
    More preciesely, single-label, multi-class classification
    """
    def __init_(self):
        pass

    def load_data(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data    
        self.test_labels = test_labels
        
    def preprocessing(self):
        """
        """
        def vectorize_sequences(sequences, dimension=10000):
            # Selected (2). Create an all-zero matrix of shape (len(sequences), dimension)
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.  # set specific indices of results[i] to 1s
            return results
        
        self.x_train = vectorize_sequences(train_data)
        self.x_test = vectorize_sequences(test_data)
        self.one_hot_train_labels = to_categorical(train_labels)
        self.one_hot_test_labels = to_categorical(test_labels)

    def build_model(self):
        """
       """
        self.model = models.Sequential()
        # set 64-dimensional hidden layer space so that 64 > 46 output classes
        self.model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(46, activation='softmax'))

        # Binary classification problem, the output of our network is a probability 
        # => best to use the binary_crossentropy loss.
        # Crossentropy is usually the best choice when dealing with models that output probabilities. 
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    def partially_validate_model(self, show=True):
        """
        """
        x_val = self.x_train[:1000]
        partial_x_train = self.x_train[1000:]
        y_val = self.one_hot_train_labels[:1000]
        partial_y_train = self.one_hot_train_labels[1000:]
        history = self.model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
        
        history_dict = history.history
        print("keys in a dictionary everything that happened during training.", history_dict.keys())
        
        def show_val_result(history):
            acc = history.history['acc']
            val_acc = history.history['val_acc']   
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
        self.model.fit(self.x_train, self.one_hot_train_labels, epochs=8, batch_size=512)

    def test_model(self):
        return self.model.evaluate(self.x_test, self.one_hot_test_labels)

    def predict(self):
        return self.model.predict(self.x_test)

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    # Downloading data from https://s3.amazonaws.com/text-datasets/reuters.npz
    # a set of short newswires and their topics.
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print("train_data.shape: ",train_data.shape)
    print("test_data.shape: ",test_data.shape)
    print("len(train_labels): ",len(train_labels))
    print("len(test_labels): ",len(test_labels))
    print("test_data[0][:10]", test_data[0][:10])
    print("train_labels[:10]", train_labels[:10])
    print("maximum number of words in train data: ", max([max(sequence) for sequence in train_data]))
    print(DecodeKerasString.decode_keras_string(reuters, single_train_data=train_data[0]))

    multiclass_classification = MulticlassClassification()
    multiclass_classification.load_data(train_data, train_labels, test_data, test_labels)
    multiclass_classification.preprocessing()
    multiclass_classification.build_model()
    multiclass_classification.partially_validate_model()
    multiclass_classification.train_model()
    test_loss, test_acc = multiclass_classification.test_model()
    print("test_acc: ",test_acc)

    pred = multiclass_classification.predict()
    print("pred_result: ", pred)