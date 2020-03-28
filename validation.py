# VALIDATION CODE
import tensorflow as tf
import time
import numpy as np
import sys
from termcolor import colored

'''
This is used to validate the model on validation data

'''
###############################################


# Function to monitor progress
def validation_progress(cnt, v_len, time_, loss):
    overall_complete = cnt / float(v_len)
    sec = int(time_) % 60
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    msg = "\r Validation_Time (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,   Validation loss: {3:s}    Overall Progress:{4:.1%}," \
          " completed {5:d} out of {6:d} logs".format(hr, mint, sec, loss, overall_complete, cnt, v_len)
    sys.stdout.write(colored(msg, 'blue'))
    sys.stdout.flush()



def validate_model(model_loc, v_len, data_set, batch_size):
    '''
    :param model_loc: DNN (Sequential) model used to validate data on
    :param v_len: length of validation data
    :param data_set: validation dataset object
    :param batch_size:  batch size used to consume data from the dataset API
    :return: loss value of validation
    '''

    cnt = 1
    start = time.time()
    loss_metric = 0
    summed_loss = []
    for (batch, (images, labels)) in (enumerate(data_set)):

        logits_out = model_loc(images)
        losses = tf.squared_difference(labels, logits_out)

        if not summed_loss:
            for i, loss in enumerate(losses):
                summed_loss.append(tf.reduce_sum(loss))
        else:
            for i, loss in enumerate(losses):
                summed_loss[i] = tf.reduce_sum(loss)
        loss_metric = tf.reduce_mean(summed_loss).numpy()

        time_ = time.time() - start
        validation_progress(cnt*batch_size, v_len, time_, loss_metric)

        cnt += 1
    return loss_metric
