import numpy as np
import tensorflow as tf
import numexpr as ne


def binaryCrossEntropy(y_true, y_pred):
    p = tf.math.reduce_mean(y_true.astype(np.float32))
    val = -tf.math.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))/(-(p*tf.math.log(p) + (1-p)*tf.math.log(1-p)))
    return val.numpy()

def softmax(x, axis):
    exp_mat = np.exp(x)
    exp_mat = ne.evaluate('exp(x)')
    return exp_mat / np.sum(exp_mat, axis=axis)

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
