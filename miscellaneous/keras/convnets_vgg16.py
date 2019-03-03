'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import keras
import os
import numpy as np
from keras import backend as K

"""
VGG16 architecture, developed by Karen Simonyan and Andrew Zisserman in 2014, 
a simple and widely used convnet architecture for ImageNet.

Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556
"""
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from util import SmoothCurve

class ConvnetsVGG16(object):
    def __init_(self):
        pass
        
    def preprocessing(self, train_dir, validation_dir, test_dir, method, batch_size=20, augment=False):
        """
            - Read the picture files.
            - Decode the JPEG content to RBG grids of pixels.
            - Convert these into floating point tensors.
            - Rescale the pixel values (between 0 and 255) to the [0, 1] interval.
        """
        # All images will be rescaled by 1./255
        if method == "extend":
            augment = True
        elif method == "record":
            augment = False
            train_features, self.train_labels = self.extract_features(train_dir, 2000, batch_size)
            validation_features, self.validation_labels = self.extract_features(validation_dir, 1000, batch_size)
            test_features, self.test_labels = self.extract_features(test_dir, 1000, batch_size)
            
            def reshape_to_dense(features, sample_count):
                #conv_base final feature map has shape (4, 4, 512)
                #to feed them to a densely-connected classifier, flatten them
                return np.reshape(features, (sample_count, 4 * 4 * 512))

            self.train_features = reshape_to_dense(train_features, 2000)
            self.validation_features = reshape_to_dense(validation_features, 1000)
            self.test_features = reshape_to_dense(test_features, 1000)
        else:
            raise Exception("model not found")

        if augment:
            train_datagen = self._augment()
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        if method == "extend":
            self.train_generator = self._get_generator(train_datagen, train_dir, batch_size)
            self.validation_generator = self._get_generator(test_datagen, validation_dir, batch_size)
            self.test_generator = self._get_generator(test_datagen, test_dir, batch_size)
    
    def _augment(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        return train_datagen

    def _get_generator(self, datagen, directory, batch_size):
        return datagen.flow_from_directory(
            directory, # Since we use binary_crossentropy loss, we need binary labels
            target_size=(150, 150), # All images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels

    def extract_features(self, directory, sample_count, batch_size):
        features = np.zeros(shape=(sample_count, 4, 4, 512))
        labels = np.zeros(shape=(sample_count))

        datagen = ImageDataGenerator(rescale=1./255)
        generator = self._get_generator(datagen, directory, batch_size)
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = self.conv_base.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                # Note that since generators yield data indefinitely in a loop,
                # we must `break` after every image has been seen once.
                break
        return features, labels

    def build_model(self, method):
        """
        VGG16

        - weights, to specify which weight checkpoint to initialize the model from
        - include_top, which refers to including or not the densely-connected classifier on top of the network. 
        - input_shape, the shape of the image tensors that we will feed to the network. 
        """
        self.conv_base = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3))
        print("Base model looks like this: \n")
        self.conv_base.summary()     
        """
        The final feature map has shape (4, 4, 512). 
        That's the feature on top of which we will stick a densely-connected classifier.
        Two possible ways to proceed:
            (1) method record: run the convolutional base over our dataset, record its output to a Numpy array on disk, 
                then using this data as input to a standalone densely-connected classifier.
                Fast, cheap computation. may overfit. no data augmentation
            (2) method extend: add Dense layers on top of conv_base
                Slow, expensive computation (GPU needed). data augmentation.
        """
        if method == "extend":
            self.model = models.Sequential()
            self.model.add(self.conv_base)
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.summary()
            """
            Freezing a layer or set of layers means preventing their weights from getting updated during training. 
            If we don't do this, then the representations that were previously learned by the convolutional base would get modified and destroyed.
            """
            print('This is the number of trainable weights, before freezing the conv base:', len(self.model.trainable_weights))
            self.conv_base.trainable = False
            # only the weights from the two Dense layers that we added will be trained
            print('This is the number of trainable weights, after freezing the conv base:', len(self.model.trainable_weights))          
            
        elif method == "record":
            self.model = models.Sequential()
            self.model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.summary()
        
        else:
            raise Exception("model not found.")

        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
            loss='binary_crossentropy',
            metrics=['acc'])

    def fine_tune_model(self):
        """
        fine-tune the last 3 convolutional layers, 
        the last 3 layers block5_conv1, block5_conv2 and block5_conv3 should be trainable.
        Why not fine-tune more layers?
            - Earlier layers in the convolutional base encode more generic, reusable features
            - The more parameters we are training, the more we are at risk of overfitting
        """
        self.conv_base.trainable = True
        set_trainable = False
        for layer in self.conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
            loss='binary_crossentropy',
            metrics=['acc'])

    def train_model(self, method, epochs, batch_size, save=False):
        if method == "extend":
            self.history = self.model.fit_generator(
                self.train_generator,
                steps_per_epoch=100,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=50)
        else:
            self.history = self.model.fit(self.train_features, self.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.validation_features, self.validation_labels))
        
        if save:
            self.model.save('cats_and_dogs_small_2.h5')

    def partially_validate_model(self, show=True, smooth=False):
        """
        """
        def show_val_result(history):
            acc = history.history['acc']
            val_acc = history.history['val_acc']   
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)
            
            if smooth:
                plt.plot(epochs, SmoothCurve.smooth_curve(loss), 'bo', label='Training loss')
                plt.plot(epochs, SmoothCurve.smooth_curve(val_loss), 'b', label='Validation loss')
                plt.plot(epochs, SmoothCurve.smooth_curve(acc), 'bo', label='Training acc')
                plt.plot(epochs, SmoothCurve.smooth_curve(val_acc), 'b', label='Validation acc')

            else:            
                plt.plot(epochs, loss, 'bo', label='Training loss')
                plt.plot(epochs, val_loss, 'b', label='Validation loss')
                plt.plot(epochs, acc, 'bo', label='Training acc')
                plt.plot(epochs, val_acc, 'b', label='Validation acc')

            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.clf()   # clear figure
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        if show == True:      
            show_val_result(self.history)
    
    def test_model(self):
        return self.model.evaluate_generator(self.test_generator, steps=50)


if __name__=="__main__":
    print("keras.__version__: ", keras.__version__)
    print("Backend TensorFlow __version__: ", K.tensorflow_backend.tf.__version__)
        
    convnets = ConvnetsVGG16()

    train_dir = os.path.join(os.path.curdir, "data", "train")
    validation_dir = os.path.join(os.path.curdir, "data", "validation")
    test_dir = os.path.join(os.path.curdir, "data", "test")
   
    """
    methods:
    (1) record: record the output of conv_base on our data 
        and using these outputs as inputs to a new model.
    (2) extend: add Dense layers on top of conv_base
    (3) no VGG16
    """
    methods = ["record", "extend"]
    for method in methods:
        print("method: ", method)
        convnets.build_model(method=method)
        convnets.preprocessing(train_dir, validation_dir, test_dir, method, batch_size=20)
        convnets.train_model(method=method, epochs=100, batch_size=20, save=False)
        convnets.partially_validate_model(show=True, smooth=False)

    #fine tune example
    print("A fine tuning example.")
    convnets.fine_tune_model()
    convnets.train_model(method="extend", epochs=100, batch_size=20, save=False)
    convnets.partially_validate_model(show=True, smooth=True)
    test_loss, test_acc = convnets.test_model()
    
