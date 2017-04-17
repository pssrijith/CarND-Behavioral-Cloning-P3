
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def build_lenet() :
    '''
    Builds a lenet arch
    [ input - > c1:5x5x6:relu -> max(2x2) -> c2:5x5x16:relu -> max(2,2) -> flat -> fc(120) -> fc(84) - fc(1) ]
    :return: Keras Sequential model
    '''
    model = Sequential()
    model.add(Cropping2D(cropping=((55,22),(0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(16,5,5, activation="relu"))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model