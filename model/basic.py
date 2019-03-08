import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import tensorflow as tf
from tensorflow import layers
from lib import files

class SequentialNN(object):
    """
    Base class for arbitrary feedforward neural network.
    methods:
    get_trainable_variables
    load_from_npz
    save_to_npz

    need to implement method:
    build_inference
    """

    def __init__(self, name, norm_mtd='default'):
        self.name = name 
        self.reuse = False
        self.norm_mtd = norm_mtd

        self.training = tf.placeholder(tf.bool, shape=[], name=name + "_training")
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name=name + "keep_prob")
        
        self.cost = self.extra_loss = 0
        self.sum_op = self.extra_sum_op = []

    def get_trainable_variables(self, update=False):
        self.vars = [v for v in tf.trainable_variables() if self.name in v.name]
        return self.vars
    
    def load_from_npz(self, path, sess):
        """
        Find parameter by name
        """
        variables = [v for v in tf.trainable_variables() if v.name.find(self.name) > -1]
        variables += [v for v in tf.global_variables() if v.name.find(self.name) > -1 and v.name.find("moving") > -1]
        files.load_and_assign_npz(sess, path, variables)
    
    def save_to_npz(self, path, sess):
        variables = [v for v in tf.trainable_variables() if self.name in v.name]
        variables += [v for v in tf.global_variables() if self.name in v.name and "moving" in v.name]
        files.save_npz(variables, path, sess)

    def print_trainble_vairables(self):
        for v in self.vars:
            print("%s\t\t\t\t%s" % (v.name, str(v.get_shape().as_list())))

    def __call__(self, input, update_collection=None):
        """
        Automatically add scope, and concat input
        """
        with tf.variable_scope(self.name):
            x = input
            if type(input) is list:
                if len(input) > 1:
                    x = tf.concat(input, len(input[0].get_shape())-1, "concat_input_gen")
                else:
                    x = input[0]
            return self.build_inference(x, update_collection) 
    
    def set_reuse(self, reuse=True):
        self.reuse = True

    def build_train_op(self, lr):
        self.vars = [v for v in tf.trainable_variables() if self.name in v.name]

        #reg_losses = sum([1e-5 * tf.reduce_sum(v ** 2) for v in self.vars])
        #with tf.name_scope(self.name):
        #    self.sum_op.append(tf.summary.scalar("regularize", reg_losses))
        #self.cost += tf.identity(reg_losses, self.name + "_regularize")
        
        if len(self.sum_op) > 0:
            self.sum_op = tf.summary.merge(self.sum_op)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)

        with tf.control_dependencies(self.update_ops):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.,
                beta2=0.9).minimize(self.cost, var_list=self.vars, colocate_gradients_with_ops=True)
