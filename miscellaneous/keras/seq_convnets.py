'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
import os
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras import preprocessing
from keras.datasets import imdb
import matplotlib.pyplot as plt

class SeqConvnets(object):
    """
    """
    def __init_(self):
        pass
    
    def set_conf(self, lookback, delay, batch_size=128, step=6):
        self.lookback = lookback
        self.delay = delay
        self.batch_size = batch_size # generator
        self.step = step
        self.input_shape_var = None

    def gen_data(self, data, train_set_size, val_set_size, test_set_size=None, reverse=False):
        if reverse == False:
            self.train_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=0,
                        max_index=train_set_size,
                        shuffle=True,
                        step=self.step, 
                        batch_size=self.batch_size)

            self.val_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=train_set_size + 1,
                        max_index=val_set_size,
                        step=self.step,
                        batch_size=self.batch_size)

            self.test_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=val_set_size + 1,
                        max_index=None,
                        step=self.step,
                        batch_size=self.batch_size)
        else:
            self.train_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=0,
                        max_index=train_set_size,
                        shuffle=True,
                        step=self.step, 
                        batch_size=self.batch_size,
                        reverse=True)

            self.val_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=train_set_size + 1,
                        max_index=val_set_size,
                        step=self.step,
                        batch_size=self.batch_size,
                        reverse=True)

            self.test_gen = self.generator(data,
                        lookback=self.lookback,
                        delay=self.delay,
                        min_index=val_set_size + 1,
                        max_index=None,
                        step=self.step,
                        batch_size=self.batch_size,
                        reverse=True)            

        # This is how many steps to draw from `val_gen`
        # in order to see the whole validation set:
        self.val_steps = (val_set_size - (train_set_size + 1) - self.lookback) // self.batch_size

        # This is how many steps to draw from `test_gen`
        # in order to see the whole test set:
        self.test_steps = (len(data) - (val_set_size + 1) - self.lookback) // self.batch_size
        
        self.input_shape_var = data.shape[-1]

        # return tuple
        return (self.train_gen, self.val_gen, self.test_gen, self.val_steps, self.test_steps)

    def generator(self, data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6, reverse=False):
        """
        - data: The original array of floating point data, which is normalized.
        - lookback: How many timesteps back should our input data go.
        - delay: How many timesteps in the future should our target be.
        - min_index and max_index: Indices in the data array that delimit which timesteps to draw from. 
        - shuffle: Whether to shuffle our samples or draw them in chronological order.
        - batch_size: The number of samples per batch.
        - step: The period, in timesteps, at which we sample data. We will set it 6 in order to draw one data point every hour
        """
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]

            if reverse == False:
                yield samples, targets
            else:
                yield samples[:, ::-1, :], targets

    def build_conv_model(self):
        """
        Combining CNNs and RNNs to process long sequences

        https://keras.io/layers/convolutional/
        1D convolution layer (e.g. temporal convolution)
        
        Not great performance but faster
        """
        self.model = models.Sequential()
        
        self.model.add(layers.Conv1D(filters=32, # the dimensionality of the output space
                                kernel_size=5, # an integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
                                activation='relu',
                                input_shape=(None, float_data.shape[-1])))
        self.model.add(layers.MaxPooling1D(3))
        self.model.add(layers.Conv1D(32, 5, activation='relu'))
        self.model.add(layers.MaxPooling1D(3))
        self.model.add(layers.Conv1D(32, 5, activation='relu'))
        self.model.add(layers.GlobalMaxPooling1D())
        self.model.add(layers.Dense(1)) # a regression problem, no activation function on the last Dense layer
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary()        

    def build_seq_conv_model(self):
        """
        Speed of convnets + order-sensitivity of RNN = Conv 1D as preprocessing before RNN 
        
        - The convnet: turn the long input sequence into much shorter (downsampled) sequences of higher-level features
        - RNN: take this extracted features as input
        
        Great for long seq, e.g. sequences with thousands of steps
        """
        K.clear_session()
        self.model = models.Sequential()
        self.model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
        self.model.add(layers.MaxPooling1D(3))
        self.model.add(layers.Conv1D(32, 5, activation='relu'))
        # add regularized GRU after Conv 1D
        self.model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
        self.model.add(layers.Dense(1)) # a regression problem, no activation function on the last Dense layer
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary()        

    def train_model(self, epochs=20, steps_per_epoch=500):
        self.history = self.model.fit_generator(self.train_gen,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                validation_data=self.val_gen,
                                validation_steps=self.val_steps)
    
    def visualize_history(self, show=True):
        history = self.history
        
        def show_val_result(history):
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(loss) + 1)
            
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        if show == True:      
            show_val_result(history)


# Outside of the class..
# --------------------
def get_data(file_path, train_set_size, show=False, normalize=True):
    """
    Read file, convert to a NumPy array
    """
    with open(file_path) as f:
        data = f.read()
        lines = data.split('\n')
        header = lines[0].split(',')
        lines = lines[1:]
        print("header    : ", header)
        print("len(lines): ", len(lines))

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    
    if show == True:
        column = float_data[:, 1] # temperature (in degrees Celsius)
        plt.plot(range(len(column)), column)
        plt.show()

    if normalize == True:
        mean_lst = float_data[:train_set_size].mean(axis=0)
        print("mean (temprature in degrees Celsius))        : ", mean_lst[1])
        float_data -= mean_lst
        std_lst = float_data[:train_set_size].std(axis=0)
        print("std (temprature in degrees Celsius))         : ", std_lst[1])
        float_data /= std_lst
    return float_data, mean_lst, std_lst


if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    file_path = os.path.join(os.path.curdir, "data", "jena_climate_2009_2016.csv")
    float_data, mean_lst, std_lst = get_data(file_path, train_set_size=200000)

    seq_convnets = SeqConvnets()
    seq_convnets.set_conf(lookback=1440, delay=144)
    seq_convnets.gen_data(data=float_data, train_set_size=200000, val_set_size=300000)
    
    # conv 1D model
    # seq_convnets.build_conv_model()
    # seq_convnets.train_model(epochs=10)
    # seq_convnets.visualize_history()

    # use conv 1D before RNN
    seq_convnets.set_conf(lookback=1440, delay=144, step=3) # look at high-resolution timeseries
    seq_convnets.gen_data(data=float_data, train_set_size=200000, val_set_size=300000)
    seq_convnets.build_seq_conv_model()
    seq_convnets.train_model(epochs=10)
    seq_convnets.visualize_history()