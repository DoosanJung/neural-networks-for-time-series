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

class AdvancedRNN(object):
    """
    1. Recurrent dropout, a specific, built-in way to use dropout to fight overfitting in recurrent layers.
    
    2. Stacking recurrent layers, to increase the representational power of the network 
    (at the cost of higher computational loads).
    
    3. Bidirectional recurrent layers, which presents the same information 
    to a recurrent network in different ways, increasing accuracy and mitigating forgetting issues.

    Problem statement and args:
    given data going as far back as `lookback` timesteps (a timestep is 10 minutes) 
    and sampled every `steps` timesteps, can we predict the temperature in `delay` timesteps?

    lookback = 720, i.e. our observations will go back 5 days.
    steps = 6, i.e. our observations will be sampled at one data point per hour.
    delay = 144, i.e. our targets will be 24 hours in the future.
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

    def build_dense_model(self):
        """
        simple and cheap model for the baseline (e.g. small densely-connected network)
        It could be easily worse than non-machine learning baseline
        """
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=(self.lookback // self.step, self.input_shape_var)))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(1)) # a regression problem, no activation function on the last Dense layer
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary()        

    def build_gru_model(self):
        """
        GRU layer, developed by Cho et al. in 2014.
        https://arxiv.org/abs/1406.1078
        
        GRU (Gated recurrent unit) layers work by leveraging the same principle as LSTM, 
        - pros: cheaper to run,
        - cons: may not have quite as much representational power as LSTM
        """
        K.clear_session()
        self.model = models.Sequential()
        self.model.add(layers.GRU(32, 
                input_shape=(None, self.input_shape_var)))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary()        

    def build_regularized_gru_model(self):
        """
        Dropout with a recurrent network, Yarin Gal in 2015.
        https://arxiv.org/abs/1506.02142

        Applying dropout before a recurrent layer hinders learning rather than helping with regularization. 
        In 2015, Yarin Gal, as part of his Ph.D. thesis on Bayesian deep learning, determined the proper way 
        to use dropout with a recurrent network.

        The same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of 
        a dropout mask that would vary randomly from timestep to timestep. Using the same dropout mask at every timestep 
        allows the network to properly propagate its learning error through time
        """
        K.clear_session()
        self.model = models.Sequential()
        self.model.add(layers.GRU(32, 
                dropout=0.2, 
                recurrent_dropout=0.2, 
                input_shape=(None, self.input_shape_var)))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary() 

    def build_stacked_regularized_gru_model(self):
        """
        As long as you are not overfitting too badly, then you are likely under-capacity
        
        Regularize => Less overfitting => able to stack => increase the capacity of the network.
        e.g. Google translate algorithm: a stack of seven large LSTM layers.
        """
        K.clear_session()
        self.model = models.Sequential()
        # 1st GRU layer
        self.model.add(layers.GRU(32, 
                dropout=0.1, 
                recurrent_dropout=0.5, 
                # all intermediate layers should return their full sequence of outputs (a 3D tensor) 
                # rather than their output at the last timestep
                return_sequences=True, 
                input_shape=(None, self.input_shape_var)))
        # 2nd GRU layer
        self.model.add(layers.GRU(64, 
                activation='relu',
                dropout=0.1,
                recurrent_dropout=0.5))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')
        self.model.summary()   

    def build_bidirectional_model(self):
        """
        A bidirectional RNN exploits this idea to improve upon the performance of chronological-order RNNs: 
        it looks at its inputs sequence both ways, obtaining potentially richer representations 
        and capturing patterns that may have been missed by the chronological-order version alone.

        Bidirectional() will create a second, separate instance of this recurrent layer, 
        and will use one instance for processing the input sequences in chronological order 
        and the other instance for processing the input sequences in reversed order. 
        """
        K.clear_session()
        self.model = models.Sequential()
        # Bidirectional layer
        self.model.add(layers.Bidirectional(
            layers.GRU(32), 
            input_shape=(None, self.input_shape_var)))
        self.model.add(layers.Dense(1))    
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mae')

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

def evaluate_naive_method(val_gen, val_steps):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    res = np.mean(batch_maes)
    return res 

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    file_path = os.path.join(os.path.curdir, "data", "jena_climate_2009_2016.csv")
    float_data, mean_lst, std_lst = get_data(file_path, train_set_size=200000)

    advanced_rnn = AdvancedRNN()
    advanced_rnn.set_conf(lookback=1440, delay=144)
    _, val_gen, _, val_steps, _ = advanced_rnn.gen_data(data=float_data, train_set_size=200000, val_set_size=300000)
    
    # a common sense, non-machine learning baseline
    res = evaluate_naive_method(val_gen, val_steps)
    print("baseline MAE                                 : {}".format(str(res)))
    print("baseline MAE (temperature in degrees Celsius): {}".format(res * std_lst[1]))

    # basic model
    advanced_rnn.build_dense_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()

    # GRU
    advanced_rnn.build_gru_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()

    # Regularize with dropout properly
    advanced_rnn.build_regularized_gru_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()

    # Stack them
    advanced_rnn.build_stacked_regularized_gru_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()

    # bidirectional - antichronical order
    # on such a text dataset, reversed-order processing works just as well as chronological processing
    # but not in this particular temparature prediction => poorly underperforming
    advanced_rnn.gen_data(data=float_data, train_set_size=200000, val_set_size=300000, reverse=True)
    advanced_rnn.build_gru_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()

    # bidirectional - layers.Bidirectional()
    advanced_rnn.gen_data(data=float_data, train_set_size=200000, val_set_size=300000)
    advanced_rnn.build_bidirectional_model()
    advanced_rnn.train_model(epochs=10)
    advanced_rnn.visualize_history()   