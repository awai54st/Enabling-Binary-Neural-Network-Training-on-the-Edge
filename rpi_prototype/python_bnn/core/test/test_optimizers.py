import numpy as np
import tensorflow as tf
from core.optimizers import Adam


def test_Adam_completely_equal():
    class TestClass:
        def __init__(self, x, dtype):
            self.dtype = dtype
            self.x = np.array(x, dtype=dtype)

        @property
        def weights(self):
            return np.array([self.x])

        @property
        def gradients(self):
            return np.array([self.x])

        def set_weights(self, w):
            self.x = w[0].astype(self.dtype)

    dtype = np.float32
    lr = 1e-3
    opt_np = Adam(lr=lr, dtype=dtype)
    opt_tf = tf.keras.optimizers.Adam(learning_rate=lr)

    z = tf.Variable(10, dtype=tf.float32)
    loss = lambda: z**2/2.0 # dl/dx=x

    x = np.array(10, dtype=dtype)
    test_layer_np = TestClass(x, dtype)

    for i in range(7):
        step_count_tf = opt_tf.minimize(loss, [z])
        step_count_np = opt_np.update([test_layer_np])
        assert step_count_tf == step_count_np
        assert (test_layer_np.x == z.numpy())

        Adam_param = list(opt_np.adam_dict.values())[0]
        Adam_tf_variables = opt_tf.variables()
        assert Adam_tf_variables[1] == Adam_param.m_adam_arr[0]
        assert Adam_tf_variables[2] == Adam_param.v_adam_arr[0]
        #assert np.allclose(Adam_tf_variables[1].numpy(), Adam_param.m_dw)
        #assert np.allclose(Adam_tf_variables[2].numpy(), Adam_param.v_dw)

        assert Adam_param.m_adam_arr[0].dtype == dtype
        assert Adam_param.v_adam_arr[0].dtype == dtype
        assert opt_np.params["beta1"].dtype == dtype
        assert opt_np.params["beta2"].dtype == dtype
        assert opt_np.params["epsilon"].dtype == dtype
        assert opt_np.params["lr"].dtype == dtype

        Adam_tf_config = opt_tf.get_config()
        assert Adam_tf_config["learning_rate"] == opt_np.params["lr"]
        assert Adam_tf_config["beta_1"] == opt_np.params["beta1"]
        assert Adam_tf_config["beta_2"] == opt_np.params["beta2"]
        assert np.float32(Adam_tf_config["epsilon"]) == opt_np.params["epsilon"]