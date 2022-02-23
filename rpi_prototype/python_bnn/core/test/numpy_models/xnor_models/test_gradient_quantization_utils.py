import numpy as np
import tensorflow as tf
from core.test.test_helper.assert_helper import assert_by_percentage

import core.numpy_models.xnor_models.gradient_quantization_utils as gu
from core.keras_models.keras_xnor_models import binarization_utils as ku


def test_FXP_quantize():
    dtype = np.float32
    
    test_value = np.random.uniform(-1, 1, (2, 10, 10, 3))
    test_point = np.random.uniform(-1, 1, (2, 10, 10, 3))

    test_shifted_value_np, test_shifted_point_np, test_shifted_width_np = gu.FXP_quantize(test_value, test_point, 4.0, dtype)
    test_shifted_value_tf, test_shifted_point_tf, test_shifted_width_tf = ku.FXP_quantize(test_value, test_point, 4.0)

    assert test_shifted_value_np.dtype == dtype
    assert test_shifted_point_np.dtype == dtype
    assert test_shifted_width_np.dtype == dtype
    assert (test_shifted_value_np == test_shifted_value_tf.numpy()).all()
    assert (test_shifted_point_np == test_shifted_point_tf.numpy()).all()
    assert (test_shifted_width_np == test_shifted_width_tf.numpy()).all()
    

def test_FXP_quantize_large_int():
    dtype = np.float32
    
    test_value = np.random.randint(-2000, 2000, (100, 16, 16, 512))

    test_shifted_value_np, test_shifted_point_np, test_shifted_width_np = gu.FXP_quantize(test_value, 4.0, 4.0, dtype)
    test_shifted_value_tf, test_shifted_point_tf, test_shifted_width_tf = ku.FXP_quantize(test_value, 4.0, 4.0)

    assert test_shifted_value_np.dtype == dtype
    assert test_shifted_point_np.dtype == dtype
    assert test_shifted_width_np.dtype == dtype
    assert (test_shifted_value_np == test_shifted_value_tf.numpy()).all()
    assert (test_shifted_point_np == test_shifted_point_tf.numpy()).all()
    assert (test_shifted_width_np == test_shifted_width_tf.numpy()).all()
    
    
def test_log2():
    dtype = np.float32
    
    np.random.seed(0)
    test_value = np.random.uniform(1e-3, 1, (2, 10, 10, 3)).astype(dtype)
    
    test_log_np = gu.log2(test_value, dtype=dtype)
    test_log_tf = ku.log2(test_value)
    
    assert test_log_np.dtype == dtype
    assert np.allclose(test_log_np, test_log_tf)
    

def test_LOG_quantize():
    dtype = np.float32
    
    np.random.seed(2)
    test_value = np.random.uniform(-1, 1, (2, 10, 10, 3))
    test_value_tf = tf.Variable(test_value, dtype=tf.float32)
    
    test_log_np = gu.LOG_quantize(test_value, 4.0)
    test_log_tf = ku.LOG_quantize(test_value_tf, 4.0)
    
    assert test_log_np.dtype == dtype
    assert (test_log_np == test_log_tf.numpy()).all()


def test_LOG_quantize_with_zeros():
    dtype = np.float32

    np.random.seed(2)
    test_value = np.random.uniform(-1, 1, (2, 10, 10, 3))
    test_value[0, 1:4, 1:4, :] = 0
    test_value_tf = tf.Variable(test_value, dtype=tf.float32)
    
    test_log_np = gu.LOG_quantize(test_value, 4.0)
    test_log_tf = ku.LOG_quantize(test_value_tf, 4.0)

    assert test_log_np.dtype == dtype
    assert (test_log_np == test_log_tf.numpy()).all()