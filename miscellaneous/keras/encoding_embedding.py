'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import numpy as np
import string
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import imdb
from keras import preprocessing


class OneHotEncoding(object):
    @staticmethod
    def word_level(samples):
        # only take into account the top-1000 most common words
        tokenizer = Tokenizer(num_words=1000)
        # This builds the word index
        tokenizer.fit_on_texts(samples)

        # This turns strings into lists of integer indices.
        sequences = tokenizer.texts_to_sequences(samples)

        # You could also directly get the one-hot binary representations.
        # Note that other vectorization modes than one-hot encoding are supported!
        one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

        # This is how you can recover the word index that was computed
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))


class WordEmbedding(object):
    """
    Obtaining word embeddings:

    1. 
    Learn word embeddings jointly with the main task you care about 
    (e.g. document classification or sentiment prediction). 
    In this setup, you would start with random word vectors, 
    then learn your word vectors in the same way that you learn the weights of a neural network.

    2.
    Load into your model word embeddings that were pre-computed using a different machine learning task 
    than the one you are trying to solve. 
    These are called "pre-trained word embeddings".
    In a separate file: "word_embedding_GloVe.py"

    Approach 1.
    """
    def __init_(self):
        pass

    def set_conf(self, n_token, embed_dim, max_features, max_len):
        # The Embedding layer takes at least two arguments:
        # the number of possible tokens, here 1000 (1 + maximum word index),
        self.n_token = n_token
        # and the dimensionality of the embeddings, here 64.        
        self.embed_dim = embed_dim
        # Number of words to consider as features
        self.max_features = max_features
        # Cut texts after this number of words 
        # (among top max_features most common words)
        self.maxlen = max_len

    def get_data(self):
        # Load the data as lists of integers.
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test    
        self.y_test = y_test
        
    def preprocessing(self):
        """
        This turns our lists of integers into 
        a 2D integer tensor of shape `(samples, maxlen)`
        """
        self.x_train = preprocessing.sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = preprocessing.sequence.pad_sequences(self.x_test, maxlen=self.maxlen)

    def build_model(self):
        """
        """
        self.model = models.Sequential()
        # We specify the maximum input length to our Embedding layer
        # so we can later flatten the embedded inputs
        self.model.add(layers.Embedding(self.max_features, self.embed_dim, input_length=self.maxlen))
        # After the Embedding layer, 
        # our activations have shape `(samples, maxlen, 8)`.

        # We flatten the 3D tensor of embeddings 
        # into a 2D tensor of shape `(samples, maxlen * 8)`
        self.model.add(layers.Flatten())

        # We add the classifier on top
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), \
            loss='binary_crossentropy', \
            metrics=['acc'])
        self.model.summary()

    def train_model(self, epochs, batch_size, **kwargs):
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, **kwargs)

    def test_model(self):
        return self.model.evaluate(self.x_test, self.y_test)

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    OneHotEncoding.word_level(samples)

    word_embed = WordEmbedding()
    word_embed.set_conf(n_token=1000, embed_dim=8, max_features=10000, max_len=20)
    word_embed.get_data()
    word_embed.preprocessing()
    word_embed.build_model()
    word_embed.train_model(epochs=10, batch_size=32, validation_split=0.2)
    print(word_embed.test_model())