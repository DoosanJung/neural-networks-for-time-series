'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
import os, shutil

class DecodeKerasString(object):
    @staticmethod
    def decode_keras_string(keras_data, single_train_data, decode=False):
        if decode == True:
            # word_index is a dictionary mapping words to an integer index
            word_index = keras_data.get_word_index()
            # We reverse it, mapping integer indices to words
            reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
            # We decode the review; note that our indices were offset by 3
            # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
            decoded_string = ' '.join([reverse_word_index.get(i - 3, '?') for i in single_train_data])
            return decoded_string
        else:
            return "No decoding"

class SmoothCurve(object):
    @staticmethod
    def smooth_curve(points, factor=0.9):
        """
        Replace each point with an exponential moving average of the previous points, to obtain a smooth curve.
        """
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

class DataPrepDogCat(object):
    @staticmethod
    def data_prep(orig_data_dir, base_dir):
        # The path to the directory where the original
        original_dataset_dir = orig_data_dir

        # Directories for our training, validation and test splits
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)

        # Directory with our training cat pictures
        train_cats_dir = os.path.join(train_dir, 'cats')
        os.mkdir(train_cats_dir)

        # Directory with our training dog pictures
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.mkdir(train_dogs_dir)

        # Directory with our validation cat pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        os.mkdir(validation_cats_dir)

        # Directory with our validation dog pictures
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.mkdir(validation_dogs_dir)

        # Directory with our validation cat pictures
        test_cats_dir = os.path.join(test_dir, 'cats')
        os.mkdir(test_cats_dir)

        # Directory with our validation dog pictures
        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.mkdir(test_dogs_dir)

        # Copy first 1000 cat images to train_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 cat images to validation_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)
            
        # Copy next 500 cat images to test_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)
            
        # Copy first 1000 dog images to train_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)
            
        # Copy next 500 dog images to validation_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)
            
        # Copy next 500 dog images to test_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)

        print('total training cat images:', len(os.listdir(train_cats_dir)))
        print('total training dog images:', len(os.listdir(train_dogs_dir)))
        print('total validation cat images:', len(os.listdir(validation_cats_dir)))
        print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
        print('total test cat images:', len(os.listdir(test_cats_dir)))
        print('total test dog images:', len(os.listdir(test_dogs_dir)))