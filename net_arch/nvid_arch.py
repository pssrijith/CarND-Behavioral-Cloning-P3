from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def build_nvidia():
    """
    Builds a Keras sequential model of the nvidia convnet arch

    [ input - > 160x320x3 -> crop(60,22)-> normalization ->
    c1:5x5x24:relu -> max(2x2) -> c2:5x5x36:relu -> max(2,2) ->  c3:5x55x48:relu ->max(2,2) ->
    c4: 3x3x64  -> c5 3x3x64 -> flat -> dropout ->  fc(120) -> dropout ->  fc(84) -> dropout -> fc(1) ]

    :return: Keras Sequential model
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 22), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(36, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(48, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model
