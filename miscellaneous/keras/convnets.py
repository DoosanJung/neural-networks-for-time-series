'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

class Convnets(object):
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
        self.train_images = self.train_images.reshape((60000, 28, 28, 1))
        self.train_images = self.train_images.astype('float32') / 255
        self.test_images = self.test_images.reshape((10000, 28, 28, 1))
        self.test_images = self.test_images.astype('float32') / 255

        # categorially encode the label
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def build_model(self):
        """
        a stack of Conv2D and MaxPooling2D layers
        then into a densely-connected classifier network
        """
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['mae'])
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    def train_model(self, epochs, batch_size):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size)

    def test_model(self):
        return self.model.evaluate(self.test_images, self.test_labels)

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
    
    convnets = Convnets()
    convnets.load_data(train_images, train_labels, test_images, test_labels)
    convnets.preprocessing()
    convnets.build_model()
    convnets.train_model(epochs=5, batch_size=64)
    test_loss, test_acc = convnets.test_model()
    print("test_acc: ",test_acc)