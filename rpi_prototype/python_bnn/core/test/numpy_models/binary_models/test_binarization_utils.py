import numpy as np
import tensorflow as tf
from core.numpy_models.binary_models.binarization_utils import binarize_weight
from core.keras_models.keras_xnor_models import binarization_utils

def test_binarize_weight():
    np.random.seed(10)
    dtype = np.float32

    test_input = np.random.uniform(-10, 10, (100, 32, 32, 512))

    res_forward_np = binarize_weight(test_input, dtype)

    z = tf.Variable(test_input, dtype = tf.float32)
    y = binarization_utils.binarize_weight(z)

    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()
    
    
def test_binarize_weight_large_int():
    np.random.seed(10)
    dtype = np.float32

    test_input = np.random.randint(-2000, 2000, (100, 32, 32, 512))

    res_forward_np = binarize_weight(test_input, dtype)

    z = tf.Variable(test_input, dtype = tf.float32)
    y = binarization_utils.binarize_weight(z)

    assert res_forward_np.dtype == dtype
    assert (res_forward_np == y.numpy()).all()