"""
Image cache
"""
import numpy as np
import tensorflow as tf

class SampleCache(object):
    def __init__(self, batch_size, iter_num):
        self.samples = []
        self.bs = batch_size
        self.iter_num = iter_num
        self.total = iter_num * batch_size

        # a time list of sample
        self.database = []
        self.batch_sizes = []
        # round drop policy: 
        self.drop_start = self.total - 1
        self.drop_batch = self.iter_num - 1

        self.batch_full = tf.placeholder(tf.int32, shape=[], name='batch_full')
        self.batch_full_sum = tf.summary.scalar("batch_full", self.batch_full)

    def __round_drop(self):
        # print("RD: %d %d" % (len(self.database[0]), self.drop_start))
        # every time add a batch and drop a batch
        for cnt in range(self.bs):
            for i in range(len(self.database)):
                self.database[i].__delitem__(self.drop_start)
            # move to previous batch end
            self.drop_start -= self.batch_sizes[self.drop_batch]
            # decrease the dropped batch
            self.batch_sizes[self.drop_batch] -= 1
            # if this batch is empty
            if self.batch_sizes[self.drop_batch] == 0:
                del self.batch_sizes[self.drop_batch]
            # move to previous batch size
            self.drop_batch -= 1
            # this batch is before front: circulate
            if self.drop_batch < 0:
                self.drop_batch = len(self.batch_sizes) - 1
                self.drop_start = len(self.database[0]) - 1
        # print("RDA: %d %d" % (len(self.database[0]), self.drop_start))
        # print(self.batch_sizes)

    def add(self, sample, global_iter):
        """
        sample: a list of np.array
        """
        if len(self.database) == 0:
            # deep copy of first sample
            for s in sample:
                self.database.append(list(s))
            self.database.append([global_iter] * self.bs)
            self.batch_sizes.append(self.bs)
            return
        
        # sample number + time
        num = len(sample)
        assert num == len(self.database) - 1
        # add item to database
        for i in range(num):
            self.database[i].extend(
                list(sample[i]) )
        # global iter
        self.database[-1].extend([global_iter] * self.bs)
        self.batch_sizes.append(self.bs)
        # round drop if length exceeded
        if len(self.database[0]) > self.total:
            self.__round_drop()

    def get(self):
        """Form a sample
        """
        indice = np.random.randint(0, len(self.database[0]), size=(self.bs,))
        num = len(self.database) - 1
        sample = []
        for i in range(num):
            data = []
            for j in indice:
                data.append(self.database[i][j])
            sample.append(np.array(data))
        return sample