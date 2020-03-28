from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import numpy as np
sep = os.sep
# Enable eager execution and configure auto growth of memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

# Create sequential DNN model
def my_model():
    '''
    :return:  a keras a sequential model
    '''

    model = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='elu', input_shape=(200, 200, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='elu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='elu'),
        layers.AveragePooling2D(10, 10),
        layers.Flatten(),
        layers.Dense(512, activation='elu'),
        layers.Dense(3, activation='relu')
    ])

    model.summary()
    model.load_weights('./checkpoints/cp-5.0-0744.ckpt')
    return model

# Instantiate the model
seq_model = my_model()

# Predict
def circle_find(input_img):

    '''
    :param input_img: input image of size 200X200
    :return: a numpy array of 3 elements (radius, xcoord, ycoord)
    '''

    # IMG_HEIGHT = np.shape(input_img)[0]
    # IMG_WIDTH = np.shape(input_img)[1]
    norm_img = np.expand_dims((input_img / 3.0).astype('float32'), axis=3)
    reshaped_image = tf.reshape(norm_img, [1,200,200, 1])
    tensor_img = tf.convert_to_tensor(reshaped_image)
    prd = seq_model(tensor_img)

    return np.squeeze(prd.numpy())