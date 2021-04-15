from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#import flearn.utils.tf_utils as tf_utils
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

import numpy as np


class FEDL(optimizer.Optimizer):
    """Implementation of Proximal Sarah, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001,hyper_learning_rate = 0.001, lamb=0.001, use_locking=False, name="FEDL"):
        super(FEDL, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._hp_lr = hyper_learning_rate
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._hp_lr_t = ops.convert_to_tensor(self._hp_lr, name="hyper_learning_rate")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "preG", self._name)
            self._zeros_slot(v, "preGn", self._name)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        hp_lr_t = math_ops.cast(self._hp_lr_t, var.dtype.base_dtype)
        preG = self.get_slot(var, "preG")
        preGn = self.get_slot(var, "preGn")
        var_update = state_ops.assign_sub(var, lr_t*(grad + hp_lr_t*preG - preGn))
        #var_update = state_ops.assign_sub(var, w)

        return control_flow_ops.group(*[var_update,])

    def set_preG(self, preG, client):
        ###with client.graph.as_default():
        all_vars = client.trainable_variables

        # make preG and all_vars looking same (in terms of shapes)
        new_preG, i, j = [], 0, 0
        for var in all_vars:
            j += np.prod(var.shape)
            new_preG.append(np.array(preG[i:j]).reshape(var.shape))
            i = j

        for variable, value in zip(all_vars, new_preG):
            v = self.get_slot(variable, "preG")
            # print(value)
            # print(variable)
            ###v.load(value, client.sess)
            v.assign(value)

    def set_preGn(self, preGn, client):
        ###with client.graph.as_default():
        all_vars = client.trainable_variables
        for variable, value in zip(all_vars, preGn):
            v = self.get_slot(variable, "preGn")
            # print(value)
            # print(variable)
            ###v.load(value, client.sess)
            v.assign(value)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "hyper_learning_rate": self._serialize_hyperparameter("hyper_learning_rate"),
            "lamb": self._serialize_hyperparameter("lamb"),
        }
