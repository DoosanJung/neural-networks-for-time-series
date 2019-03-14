'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
import string
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras import preprocessing
from keras.datasets import imdb
import matplotlib.pyplot as plt

class SimpleRNN(object):
    """
    """
    def __init_(self):
        pass

    def set_conf(self, max_features, max_len):
        self.max_features = max_features
        self.maxlen = max_len

    def get_data(self):
        # Load the data as lists of integers.
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')    
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test    
        self.y_test = y_test
        
    def preprocessing(self):
        """
        """
        print('Pad sequences (samples x time)')
        self.x_train = preprocessing.sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = preprocessing.sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)

    def build_model(self, lstm=False):
        """
        output types:
        
        1. The full sequences of successive outputs for each timestep 
        (a 3D tensor of shape (batch_size, timesteps, output_features)), 
        
        2. Only the last output for each input sequence 
        (a 2D tensor of shape (batch_size, output_features)). 
        
        <= controlled by the return_sequences constructor argument.
        (default value == False)
        """
        self.model = models.Sequential()

        # input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
        # output_dim: int >= 0. Dimension of the dense embedding        
        self.model.add(layers.Embedding(input_dim=10000, output_dim=32))
        
        if lstm:
            #layers.LSTM: Long Short-Term Memory layer - Hochreiter 1997.
            self.model.add(layers.LSTM(units=32, return_sequences=True))
            self.model.add(layers.LSTM(units=32, return_sequences=True))
            self.model.add(layers.LSTM(units=32))
            
        else:
            #layers.SimpleRNN: Fully-connected RNN where the output is to be fed back to input.
            #units: Positive integer, dimensionality of the output space.
            #https://keras.io/layers/recurrent/
            self.model.add(layers.SimpleRNN(units=32, return_sequences=True))
            self.model.add(layers.SimpleRNN(units=32, return_sequences=True))
            self.model.add(layers.SimpleRNN(units=32))

        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), \
            loss='binary_crossentropy', \
            metrics=['acc'])
        self.model.summary()

    def train_model(self, epochs, batch_size, **kwargs):
        self.history = self.model.fit(self.x_train, self.y_train, \
            epochs=epochs, batch_size=batch_size, **kwargs)
    
    def visualize_history(self, show=True):
        """
        """
        history = self.history
        
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
            plt.ylabel('acc')
            plt.legend()

            plt.show()
        if show == True:      
            show_val_result(history)

    def test_model(self):
        return self.model.evaluate(self.x_test, self.y_test)

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    simple_rnn = SimpleRNN()
    simple_rnn.set_conf(max_features=10000, max_len=500)
    simple_rnn.get_data()
    simple_rnn.preprocessing()
    simple_rnn.build_model()
    simple_rnn.train_model(epochs=1, batch_size=128, validation_split=0.2)
    print(simple_rnn.test_model())

    simple_rnn.build_model(lstm=True)
    simple_rnn.train_model(epochs=10, batch_size=128, validation_split=0.2)
    print(simple_rnn.test_model())