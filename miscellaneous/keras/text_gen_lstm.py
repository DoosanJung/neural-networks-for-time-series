'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import os
import sys
import random
import numpy as np
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers

class TextGenLSTM(object):
    """
    """
    def __init_(self):
        pass

    def get_data(self, data):
        """load the text into the model"""
        self.data = data

    def set_conf(self, maxlen, step, temperatures, output_len, batch_size=128):
        self.maxlen = maxlen # length of extracted character sequences
        self.step = step # sample a new sequence every `step` characters
        self.temperatures = temperatures
        self.output_len = output_len
        self.batch_size = batch_size

    def preprocessing(self):
        """
        - Extract partially-overlapping sequences of length maxlen, one-hot encode them and pack them 
        in a 3D Numpy array x of shape (sequences, maxlen, unique_characters)
        - Prepare y targets: the one-hot encoded characters that come right after each sequence
        """
        self.chars = [] # Unique characters in the corpus
        self.char_indices = {} # Dictionary mapping unique characters to their index
        self.sentences = [] # This holds our extracted sequences        
        self.next_chars = [] # This holds the targets (the follow-up characters)

        for i in range(0, len(self.data) - self.maxlen, self.step):
            self.sentences.append(self.data[i: i + self.maxlen])
            self.next_chars.append(self.data[i + self.maxlen])
        print('Number of sequences:', len(self.sentences))
        
        # List of unique characters in the corpus
        self.chars = sorted(set(self.data))
        print('Unique characters:', len(self.chars))
        
        # Dictionary mapping unique characters to their index in `self.chars`
        self.char_indices = dict((char, self.chars.index(char)) for char in self.chars)

        # Next, one-hot encode the characters into binary arrays.
        print('Vectorization...')
        self.x = np.zeros((len(self.sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[self.next_chars[i]]] = 1
        
        # first char in first sentence: p from 'preface\n\n\nsupposing that truth is a woman--what then? is the'
        for item in zip(self.chars, self.x[0][0]):
            print(item)
        print("It should contain ('p', True)")

    def custom_train_model(self, epochs):
        """
        Generate new text by repeatedly:
            1) Drawing from the model a probability distribution over the next character given the text available so far
            2) Reweighting the distribution to a certain "temperature"
            3) Sampling the next character at random according to the reweighted distribution
            4) Adding the new character at the end of the available text
        """
        for epoch in range(1, epochs):
            print('Custom epoch: ', epoch)
            self._train_model()

    def _train_model(self):
        """
        """
        # Fit the model for 1 epoch on the available training data
        self.model.fit(self.x, self.y,
                batch_size=self.batch_size,
                epochs=1)

        # Select a text seed at random
        start_index = random.randint(0, len(self.data) - self.maxlen - 1)
        generated_text = self.data[start_index: start_index + self.maxlen]
        print('--- Generating with seed: "' + generated_text + '"')

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)

            # We generate self.output_len characters
            for i in range(self.output_len):
                sampled = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(sampled, verbose=0)[0]
                next_index = self._sample(preds, temperature)
                next_char = self.chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def _sample(self, preds, temperature):
        """
        - Reweight the original probability distribution coming out of the model
        - Draw a character index from it
        """
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.exp(np.log(preds) / temperature)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def build_model(self):
        """
        a single LSTM layer + Dense classifier w/ softmax
        """
        self.model = models.Sequential()
        self.model.add(layers.LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(layers.Dense(len(self.chars), activation='softmax')) # 
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy')
        self.model.summary()        
  

# Outside of the class..
# --------------------
def get_data_from_s3():
    """
    - Using Keras.util.get_file()
    """
    file_path = keras.utils.get_file(
        'nietzsche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(file_path).read().lower()
    print('type(text)', type(text)) # str
    print('Corpus length:', len(text))
    return text


if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    text = get_data_from_s3()
    temperatures = [0.2, 0.5, 1.0, 1.2]

    text_gen_lstm = TextGenLSTM()
    text_gen_lstm.set_conf(maxlen=60, step=3, temperatures=temperatures, output_len=400)
    text_gen_lstm.get_data(data=text)
    text_gen_lstm.preprocessing()
    text_gen_lstm.build_model()
    text_gen_lstm.custom_train_model(epochs=10)