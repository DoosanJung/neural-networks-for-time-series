'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import os, shutil
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
# keras.preprocessing.image.ImageDataGenerator
# Python generators that can automatically turn image files on disk 
# into batches of pre-processed tensors
from keras.preprocessing.image import ImageDataGenerator

from util import DataPrepDogCat

class ConvnetsFromImageGen(object):
    def __init_(self):
        pass

    def augment(self):
        """
            https://keras.io/preprocessing/image/
            - rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures.
            - width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
            - shear_range is for randomly applying shearing transformations.
            - zoom_range is for randomly zooming inside pictures.
            - horizontal_flip is for randomly flipping half of the images horizontally. relevant when there are no assumptions of horizontal asymmetry (e.g. real-world pictures).
            - fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
            
            => The inputs that it sees are still heavily intercorrelated, since they come from a small number of original images
            
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        return train_datagen
                    
    def preprocessing(self, train_dir, validation_dir, augment=False):
        """
            - Read the picture files.
            - Decode the JPEG content to RBG grids of pixels.
            - Convert these into floating point tensors.
            - Rescale the pixel values (between 0 and 255) to the [0, 1] interval.
        """
        # All images will be rescaled by 1./255
        if augment:
            train_datagen = self.augment()
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=32,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    def build_model(self):
        """
        a stack of Conv2D and MaxPooling2D layers
        then into a densely-connected classifier network
        """
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
            loss='binary_crossentropy',
            metrics=['acc'])

    def train_model(self, epochs, save=False):
        # installed pillow: pip install pillow-5.4.1
        # https://keras.io/models/sequential/#fit_generator
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=50)
        if save:
            self.model.save('cats_and_dogs_small_2.h5')

    def partially_validate_model(self, show=True):
        """
        """
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
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        if show == True:      
            show_val_result(self.history)

def data_prep_func(needed=False):
    if needed:
        # The path to the directory where the original
        orig_data_dir = os.path.join(os.path.curdir, "kaggle_original_data")
        # The directory where we will store our smaller dataset
        base_dir = os.path.join(os.path.curdir, "data")
        DataPrepDogCat.data_prep(orig_data_dir=orig_data_dir, base_dir=base_dir)


if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
    
    data_prep_func(needed=False)
    
    convnets = ConvnetsFromImageGen()

    train_dir = os.path.join(os.path.curdir, "data", "train")
    validation_dir = os.path.join(os.path.curdir, "data", "validation")
    convnets.preprocessing(train_dir, validation_dir, augment=True)

    convnets.build_model()
    convnets.train_model(epochs=100, save=True)
    convnets.partially_validate_model(show=True)