'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
from keras import backend as K
from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from util import SmoothCurve

class Regression(object):
    """
    More preciesely, scalar regression
    """
    def __init_(self):
        pass

    def load_data(self, train_data, train_targets, test_data, test_targets):
        self.train_data = train_data
        self.train_targets = train_targets
        self.test_data = test_data    
        self.test_targets = test_targets
        
    def preprocessing(self):
        """
        Normalization
        """
        def normalize(data, **kwargs):
            mean = kwargs.pop("mean", [None]*5)
            std = kwargs.pop("std", [None]*5)
            if kwargs:
                raise Exception()
            
            if not all(mean):
                mean = data.mean(axis=0)
                std = data.std(axis=0)  
            return ((data - mean)/std), mean, std

        self.train_data, train_mean, train_std = normalize(self.train_data)
        self.test_data, _ ,_ = normalize(self.test_data, mean=train_mean, std=train_std)

    def build_model(self):
        """
        """
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.train_data.shape[1],)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['mae'])

    def partially_validate_model(self, fold, num_epochs, show=True):
        """
        implementing K-fold Cross Validation
        """
        k = fold
        num_val_samples = len(self.train_data) // k
        num_epochs = num_epochs
        self.all_scores = []
        self.all_mae_histories = []
        for i in range(k):
            print('processing fold #', i)
            # Prepare the validation data: data from partition # k
            val_data = self.train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.train_targets[i * num_val_samples: (i + 1) * num_val_samples]

            # Prepare the training data: data from all other partitions
            self.partial_train_data = np.concatenate(
                [self.train_data[:i * num_val_samples],
                self.train_data[(i + 1) * num_val_samples:]],
                axis=0)
            self.partial_train_targets = np.concatenate(
                [self.train_targets[:i * num_val_samples],
                self.train_targets[(i + 1) * num_val_samples:]],
                axis=0)

            # Train the model (in silent mode, verbose=0)
            history = self.model.fit(self.partial_train_data, self.partial_train_targets, epochs=num_epochs, batch_size=1)
                    
            # Evaluate the model on the validation data
            val_mse, val_mae = self.model.evaluate(val_data, val_targets, verbose=0)
            self.all_scores.append(val_mae)
            mae_history = history.history['mean_absolute_error']
            self.all_mae_histories.append(mae_history)

        print("all_scores", self.all_scores)
        print("avg_all_scores", np.mean(self.all_scores))
        average_mae_history_lst = [np.mean([x[i] for x in self.all_mae_histories]) for i in range(num_epochs)]
        smooth_mae_history_lst = SmoothCurve.smooth_curve(average_mae_history_lst[10:])
        
        def show_val_result(history_lst):
            plt.plot(range(1, len(history_lst) + 1), history_lst)
            plt.xlabel('Epochs')
            plt.ylabel('Validation MAE')
            plt.show()

        if show == True:      
            show_val_result(average_mae_history_lst)
            show_val_result(smooth_mae_history_lst)

    def train_model(self, epochs, batch_size):
        self.model.fit(self.train_data, self.train_targets, epochs=epochs, batch_size=batch_size)

    def test_model(self):
        return self.model.evaluate(self.test_data, self.test_targets)

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    # Downloading data from https://s3.amazonaws.com/keras-datasets/boston_housing.npz
    """
    The 13 features:
    Per capita crime rate.
    Proportion of residential land zoned for lots over 25,000 square feet.
    Proportion of non-retail business acres per town.
    Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    Nitric oxides concentration (parts per 10 million).
    Average number of rooms per dwelling.
    Proportion of owner-occupied units built prior to 1940.
    Weighted distances to five Boston employment centres.
    Index of accessibility to radial highways.
    Full-value property-tax rate per $10,000.
    Pupil-teacher ratio by town.
    1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
    % lower status of the population.
    """
    (train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
    print("train_data.shape: ",train_data.shape)
    print("test_data.shape: ",test_data.shape)
    print("len(train_targets): ",len(train_targets))
    print("len(test_targets): ",len(test_targets))
    print("train_targets[:10]", train_targets[:10]) 

    regression = Regression()
    regression.load_data(train_data, train_targets, test_data, test_targets)
    regression.preprocessing()
    regression.build_model()
    regression.partially_validate_model(fold=4, num_epochs=400)

    regression.train_model(epochs=80, batch_size=16)
    test_mse_score, test_mae_score = regression.test_model()
    print("test_mae_score: ",test_mae_score)