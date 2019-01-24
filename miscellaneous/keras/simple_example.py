'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
from keras import backend as K
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

class SimpleExample(object):
    def __init_(self):
        pass

    def load_data(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images    
        self.test_labels = test_labels
        
    def preprocessing(self):
        # preprocess our data by reshaping it into the shape that the self.network expects, 
        # and scaling it so that all values are in the [0, 1] interval.
        self.train_images = self.train_images.reshape((len(self.train_labels), 28 * 28))
        self.train_images = self.train_images.astype('float32') / 255
        self.test_images = self.test_images.reshape((len(self.test_labels), 28 * 28))
        self.test_images = self.test_images.astype('float32') / 255
        # categorially encode the label
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def build_train_model(self):
        self.network = models.Sequential()
        # fully connected (dense) two layers
        self.network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        self.network.add(layers.Dense(10, activation='softmax'))
        self.network.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        self.network.fit(self.train_images, self.train_labels, epochs=5, batch_size=128)

    def test_model(self):
        return self.network.evaluate(self.test_images, self.test_labels)


if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)

    # Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print("train_images.shape: ",train_images.shape)
    print("test_images.shape: ",test_images.shape)
    print("len(train_labels): ",len(train_labels))
    print("len(test_labels): ",len(test_labels))
    print("train_labels[:10]", train_labels[:10])

    first_example = SimpleExample()
    first_example.load_data(train_images, train_labels, test_images, test_labels)
    first_example.preprocessing()
    first_example.build_train_model()
    test_loss, test_acc = first_example.test_model()
    print("test_acc: ",test_acc)