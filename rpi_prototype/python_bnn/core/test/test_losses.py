import numpy as np
import tensorflow as tf
from core.test.test_helper.assert_helper import assert_by_percentage

from core.losses import CrossEntropy
from core.numpy_models.full_precision_models.layers import Softmax

from core.label_utils import label_binerizer

def test_Softmax_CrossEntropy():
    dtype = np.float32
    logits = np.array([[4.0, 2.0, 1.0], [1.0, 5.0, 1.0]], dtype=dtype)
    labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    # numpy implementation
    test_CrossEntropy = CrossEntropy(dtype=dtype)

    res_cross_entropy = test_CrossEntropy.forward(logits, labels)
    cross_entropy_grad = test_CrossEntropy.backprop()

    # tensorflow implementation
    logits_tf = tf.Variable([[4.0, 2.0, 1.0], [1.0, 5.0, 1.0]], dtype=tf.float32)
    labels_tf = tf.Variable([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    test_input = np.random.uniform(0, 1, (5, 10, 10, 3))

    with tf.GradientTape() as tape:
        y = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
    res_grads_tf = tape.gradient(y, logits_tf)


    assert np.allclose(res_cross_entropy, y.numpy())
    assert np.allclose(cross_entropy_grad, res_grads_tf.numpy())
    

    
def test_Softmax_CrossEntropy_large():
    np.random.seed(100)
    dtype = np.float32
    sample = 1000
    n_class = 10

    logits = np.random.uniform(-5, 5, size=(sample,n_class)).astype(dtype)
    labels = label_binerizer(np.random.choice(np.arange(0, n_class-1), size=(sample)), n_class=n_class)

    # numpy implementation
    test_CrossEntropy = CrossEntropy(dtype=dtype)

    res_cross_entropy = test_CrossEntropy.forward(logits, labels)
    cross_entropy_grad = test_CrossEntropy.backprop()

    # tensorflow implementation
    logits_tf = tf.Variable(logits, dtype=tf.float32)
    labels_tf = tf.Variable(labels, dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
    res_grads_tf = tape.gradient(y, logits_tf)

    assert np.allclose(res_cross_entropy, y.numpy())
    assert np.allclose(cross_entropy_grad, res_grads_tf.numpy())
    
def test_Softmax_CrossEntropy_small_int():
    np.random.seed(30)
    dtype = np.float32
    sample = 2
    n_class = 10

    logits = np.random.randint(-20, 20, size=(sample,n_class)).astype(dtype)
    labels = label_binerizer(np.random.choice(np.arange(0, n_class-1), size=(sample)), n_class=n_class)

    # numpy implementation
    test_CrossEntropy = CrossEntropy(dtype=dtype)

    res_cross_entropy = test_CrossEntropy.forward(logits, labels)
    cross_entropy_grad = test_CrossEntropy.backprop()

    # tensorflow implementation
    logits_tf = tf.Variable(logits, dtype=tf.float32)
    labels_tf = tf.Variable(labels, dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
    res_grads_tf = tape.gradient(y, logits_tf)

    assert np.allclose(res_cross_entropy, y.numpy())
    assert np.allclose(cross_entropy_grad, res_grads_tf.numpy())
    

def test_Softmax_CrossEntropy_large_int():
    np.random.seed(30)
    dtype = np.float32
    sample = 2
    n_class = 10

    logits = np.random.randint(-50, 50, size=(sample,n_class)).astype(dtype)
    labels = label_binerizer(np.random.choice(np.arange(0, n_class-1), size=(sample)), n_class=n_class)

    # numpy implementation
    test_CrossEntropy = CrossEntropy(dtype=dtype)
    test_Softmax = Softmax(dtype=dtype)

    res_cross_entropy = test_CrossEntropy.forward(logits, labels)
    cross_entropy_grad = test_CrossEntropy.backprop()

    # tensorflow implementation
    logits_tf = tf.Variable(logits, dtype=tf.float32)
    labels_tf = tf.Variable(labels, dtype=tf.float32)

    with tf.GradientTape() as tape:
        y = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
    res_grads_tf = tape.gradient(y, logits_tf)

    assert np.allclose(res_cross_entropy, y.numpy())
    assert np.allclose(cross_entropy_grad, res_grads_tf.numpy())