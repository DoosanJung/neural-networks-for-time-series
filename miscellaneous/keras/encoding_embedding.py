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

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    OneHotEncoding.word_level(samples)