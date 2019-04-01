"""
The most fundamental trainer, for configuration of runtime environment.
"""
import tensorflow as tf

import os

class BaseTrainer(object):
    """
    The base trainer class, should not be instantiated directly.
    """
    def __init__(self, step_sum_op, int_sum_op, dataloader, FLAGS, is_debug=False):
        """
        Args:
        FLAGS   :   TF app's FLAGS.
        TFLAGS  :   Customized Training FLAGS.
        gpu_mem :   'FULL' for occupying the whole GPU. "PARTIAL" for allow growth mode.
        """
        self.dataloader = dataloader
        self.FLAGS = FLAGS
        self.num_iter = FLAGS.num_iter
        self.batch_size = FLAGS.batch_size
        self.step_sum_op = step_sum_op
        self.int_sum_op = int_sum_op
        self.is_init = False

        self.has_fixed_feed = False
        self.is_debug = is_debug

        self.feed = {}
        self.fixed_feed = {}

        self.finished = False
        self.req_inter_sum = False

    def init_training(self):
        """
        Start runtime environment
        """
        print("=> Configure runtime environment")
        self.__init_runtime_environ()
        self.reset_train_state()
        self.dataloader.sess = self.sess
        if self.is_init is False:
            self.sess.run([tf.global_variables_initializer()])
            self.is_init = True
        elif self.FLAGS.reload:
            self.reload()
        elif self.FLAGS.resume:
            self.saver.restore(self.sess, self.FLAGS.train_dir)
            
    def build_train_op(self):
        """
        Build training op. When there is multiple model, building op need to be customized.
        """
        pass

    def decode_placeholders(self):
        """
        Decode according to specific trainer
        """
        pass

    def reset_train_state(self):
        """
        Called when a new training is started. Inherit for added training recording vars.
        """
        self.finished = False
        self.global_iter = 0
        self.epoch_count = 0
        self.feed = {}

    def __init_runtime_environ(self):
        """
        Init a runtime environment.
        """
        self.config = tf.ConfigProto()
        # Set GPU memory usage
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = False
        # Allow soft placement
        # self.config.allow_soft_placement=True
        # Make experiment storage dir
        os.system("mkdir " + self.FLAGS.train_dir)
        # Make summary writer
        self.summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1,
        pad_step_number=True, keep_checkpoint_every_n_hours=24.0)
        print("=> Start session.")
        self.sess = tf.Session(config=self.config)
        print("=> Session inited.")

        if self.FLAGS.debug:
            print("=> Start debugging")
            from tensorflow.python import debug as tf_debug
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    def train(self):
        """
        Init vars, write graph to summary and call epoch-wise trainer.
        """
        with self.sess:
            print("=> Initialize training.")

            self.summary_writer.add_graph(self.sess.graph)

            while self.global_iter < self.num_iter:
                self.train_epoch()
        
        self.finished = True
    
    def train_epoch(self):
        """
        Custom w.r.t models and data sampler.
        """
        pass
    
    def schedule_lr(self):
        """
        Schedule learning rate and put it into self.feed.
        """
        pass

    def schedule_weight(self):
        """
        Schedule weight and put it into self.feed.
        """
        pass
    
    def make_feed(self, sample):
        """
        Make feed dic from sample.
        """
        pass
    
    def make_fixed_feed(self):
        """
        Make fixed feed dic for intervaled summary.
        """
        pass
    
    def resume(self):
        # TODO: resume should happen before graph construction
        pass