from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import time
import numpy as np
from termcolor import colored
import sys
import validation

sep = os.sep

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

home_dir =os.getcwd()+sep

# Function used to print progress of training
def print_progress(cnt, overall, time_, loss, epoch):
    '''
    :param cnt: completed iterations
    :param overall: total number of iterations
    :param time_: time counter
    :param loss: loss value
    :param epoch: training epoch
    :return: Nothing!
    '''

    overall_complete = cnt/ float(overall - 1)
    epoch+=1
    sec = int(time_ % 60)
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    msg = "\r Time_lapsed (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,  Epoch {3:}  Training loss: {4:s}    Overall Progress:{5:.1%}," \
          " completed {6:d} out of {7:d} items".format(hr, mint, sec,epoch,loss, overall_complete, cnt, overall)
    sys.stdout.write(colored(msg, 'green'))
    sys.stdout.flush()


# Function to log data to Tensorboard summaries
def write_summaries(loss, i, global_step, vars_loc, grads_loc, train=True):
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            if train:
                tf.contrib.summary.scalar("train_loss", loss, step=global_step)
                tf.contrib.summary.scalar("step", i, step=global_step)
                tf.contrib.summary.histogram("weights", vars_loc, step=global_step)
                tf.contrib.summary.histogram("gradients", grads_loc, step=global_step)
            else:
                tf.contrib.summary.scalar("val_loss", loss, step=global_step)


# Load Data
train_images = np.load(home_dir+ r'\training_images.npy')
train_labels = np.load(home_dir+ r'\labels.npy').astype('float32')
test_images = np.load(home_dir+ r'\testing_images.npy')
test_labels =np.load(home_dir+ r'\tlabels.npy').astype('float32')
validate_images =np.load(home_dir+ r'\validation_images.npy')
validate_labels =np.load(home_dir+ r'\vlabels.npy').astype('float32')

# Get length of datasets
t_len = len(test_labels)
tr_len = len(train_labels)
v_len = len(validate_labels)


# Normalize data - divide by the largest number found in the dataset which is 3.0
# Reshape the input data demnision in order to feed it to the DNN model
train_images = np.expand_dims((train_images/3.0).astype('float32'),axis=3)
test_images = np.expand_dims((test_images / 3.0).astype('float32'),axis=3)
validate_images = np.expand_dims((validate_images /3.0).astype('float32'),axis=3)


# Build DATASET Pipeline
Buffer_size = 256
Batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(Buffer_size).batch(Batch_size)
train_dataset = train_dataset.prefetch(Buffer_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.shuffle(Buffer_size).batch(Batch_size)
test_dataset = test_dataset.prefetch(Buffer_size)

validation_dataset = tf.data.Dataset.from_tensor_slices((validate_images, validate_labels))
validation_dataset = validation_dataset.shuffle(Buffer_size).batch(Batch_size)
validation_dataset = validation_dataset.prefetch(Buffer_size)

# For data logging and checkpoints - Tensorboard
data_format = 'channels_last'
optimizer = tf.train.AdamOptimizer()
logdir = home_dir + sep +r'tensorboard_reporting' + sep
checkpont_path = home_dir + sep + r"checkpoints" + sep + "cp-{ACC:2.1f}-{log:04d}.ckpt"
checkpont_dir = os.path.dirname(checkpont_path)
summary_writer = tf.contrib.summary.create_file_writer(logdir)


IMG_HEIGHT = np.shape(train_images[0])[0]
IMG_WIDTH = np.shape(train_images[0])[1]


# Build a sequential model
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='elu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='elu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='elu'),
    layers.AveragePooling2D(10,10),
    layers.Flatten(),
    layers.Dense(512, activation='elu'),
    layers.Dense(3, activation='relu')
])

model.summary()


# Training loop
val_step = 0
loss_m = 100000
Epochs = 100
start_time = time.time()
summed_loss = []
loss_metric = 0
for epoch in range(Epochs):
    for (batch, (images, labels)) in (enumerate(train_dataset)):
        batch += 1
        step = tf.train.get_or_create_global_step()
        # Perform forward and backward passes and monitor gradients
        with tf.GradientTape() as tape:
            logits_out = model(images)
            # Loss function:  minimize (rad' - rad)^2 + (ycoord' - ycoord)^2 + (xcoord' - xcoord)^2  || xcoord, ycoord,rad >=0
            losses = tf.squared_difference(labels, logits_out)

            if not summed_loss:
                for i, loss in enumerate(losses):
                    summed_loss.append(tf.reduce_sum(loss))
            else:
                for i, loss in enumerate(losses):
                    summed_loss[i] = tf.reduce_sum(loss)

            watched_vars = tape.watched_variables()

        # Compute and apply gradients
        grads = tape.gradient(summed_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=step)
        loss_metric = tf.reduce_mean(summed_loss).numpy()
        weight_list = model.weights
        time_ = time.time() - start_time

        # print progress and write to tensorboard summaries
        if batch % 8 == 0:
            print_progress(batch * Batch_size, tr_len, time_, loss_metric, epoch)
            write_summaries(loss_metric, batch, step, weight_list[0], grads[0], train=True)
            print('\n')
    print('\n')
    # cross validate
    loss_metric_v = validation.validate_model(model, v_len, validation_dataset, Batch_size)
    write_summaries(loss_metric_v, 0, val_step, 0, 0, train=False)
    val_step += 1
    # Save model parameters
    if loss_metric_v < loss_m:
        loss_m = loss_metric_v
        model.save_weights(checkpont_path.format(ACC=loss_metric_v, log=epoch))

