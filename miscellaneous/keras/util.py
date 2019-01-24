'''
* Francois Chollet, 2017, "Deep Learning with Python"
* Francois Chollet's example code([GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks))
* I bought this book. I modified the example code a bit to confirm my understanding.
'''
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