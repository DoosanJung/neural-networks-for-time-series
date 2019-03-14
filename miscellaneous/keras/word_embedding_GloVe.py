'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import os
import numpy as np
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import optimizers
from keras import preprocessing
import matplotlib.pyplot as plt

class WordEmbeddingGloVe(object):
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
    In a separate file: "word_embed_gloveding_glove.py"

    Approach 2.
    """
    def __init_(self):
        pass

    def set_conf(self, embed_dim, max_features, max_len, training_samples, validation_samples):
        # and the dimensionality of the embeddings, here 64.        
        self.embed_dim = embed_dim
        # Number of words to consider as features
        self.max_features = max_features
        # Cut texts after this number of words 
        # (among top max_features most common words)
        self.maxlen = max_len
        # We will be training on 200 samples
        self.training_samples = training_samples
        # We will be validating on 10000 samples
        self.validation_samples = validation_samples

    def preprocessing(self, labels, texts):
        """
        This turns our lists of integers into 
        a 2D integer tensor of shape `(samples, maxlen)`
        """
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        self.data = pad_sequences(sequences, maxlen=self.maxlen)
        self.labels = np.asarray(labels)
        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', self.labels.shape)

        # Split the data into a training set and a validation set
        # But first, shuffle the data, since we started from data
        # where sample are ordered (all negative first, then all positive).
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def preprocessing_test(self, labels, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.x_test = pad_sequences(sequences, maxlen=self.maxlen)
        self.y_test = np.asarray(labels)

    def get_data(self):
        self.x_train = self.data[:self.training_samples]
        self.y_train = self.labels[:self.training_samples]
        self.x_val = self.data[self.training_samples: self.training_samples + self.validation_samples]
        self.y_val = self.labels[self.training_samples: self.training_samples + self.validation_samples]        

    def get_embedding_matrix(self, embeddings_index):
        self.embedding_matrix = np.zeros((self.max_features, self.embed_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < self.max_features:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

    def build_model(self, **kwargs):
        """
        """
        load_model = kwargs.pop("load_model", False)
        if kwargs:
            raise Exception("unexpected kwargs!")
        self.model = models.Sequential()
        self.model.add(layers.Embedding(self.max_features, self.embed_dim, input_length=self.maxlen))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        

        #load the model
        if load_model:
            self._load_model()

        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), \
            loss='binary_crossentropy', \
            metrics=['acc'])
        self.model.summary()

    def _load_model(self):
        self.model.layers[0].set_weights([self.embedding_matrix])
        self.model.layers[0].trainable = False

    def train_model(self, epochs, batch_size, **kwargs):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, \
            batch_size=batch_size, validation_data=(self.x_val, self.y_val))
        self.model.save_weights('pre_trained_glove_model.h5')

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
        self.model.load_weights('pre_trained_glove_model.h5')
        return self.model.evaluate(self.x_test, self.y_test)

def get_imdb(data_dir, train_or_test):
    if train_or_test == "train":
        dir = os.path.join(imdb_dir, 'train')
    elif train_or_test == "test":
        dir = os.path.join(imdb_dir, 'test')
    
    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname)) as f:
                    texts.append(f.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return labels, texts

def get_GloVe(data_dir):
    embeddings_index = {}
    with open(os.path.join(data_dir, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)

    imdb_dir = os.path.join(os.path.curdir, "data", "aclImdb")
    train_labels, train_texts = get_imdb(data_dir=imdb_dir, train_or_test="train")
    test_labels, test_texts = get_imdb(data_dir=imdb_dir, train_or_test="test")

    GloVe_dir = os.path.join(os.path.curdir, "data", "GloVeData")
    embeddings_index = get_GloVe(GloVe_dir)

    word_embed_glove = WordEmbeddingGloVe()
    training_samples_lst = [200, 1000, 2000, 5000]
    for training_samples in training_samples_lst:
        word_embed_glove.set_conf(embed_dim=100, \
            max_features=10000, \
            max_len=100, \
            training_samples=training_samples, \
            validation_samples=10000)
        word_embed_glove.preprocessing(train_labels, train_texts)
        word_embed_glove.get_data()
        word_embed_glove.get_embedding_matrix(embeddings_index)
        word_embed_glove.build_model(load_model=True)
        word_embed_glove.train_model(epochs=10, batch_size=32)
        word_embed_glove.visualize_history()
        word_embed_glove.preprocessing_test(test_labels, test_texts)
        test_loss, test_acc = word_embed_glove.test_model()
        print("test_acc: ",test_acc)