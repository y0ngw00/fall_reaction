# The MIT License

# Copyright (c) 2017 OpenAI (http://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import pickle
from IPython import embed
import tensorflow as tf
 
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4
        self.epsilon = 1e-8
        self.clip = 10

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        mean, var, count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        if np.isnan(np.sum(mean)) or np.isnan(np.sum(var)) or np.isnan(np.sum(count)):
            print("Nan occur")
            embed()
        else:
            self.mean = mean
            self.var = var
            self.count = count


    def apply(self, x):
        self.update(x)

        x = np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
        return x

    def save(self, path):
        data = {'mean':self.mean, 'var':self.var, 'count':self.count}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
                data = pickle.load(f)
                self.mean = data['mean']
                self.var = data['var']
                self.count = data['count']
        
    def setNumStates(self, size):
        if size != self.mean.shape[0]:
            l = size - self.mean.shape[0]
            m_new = np.zeros(l, 'float64')
            v_new = np.ones(l, 'float64')
            self.mean = np.concatenate((self.mean, m_new), axis=0)
            self.var = np.concatenate((self.var, v_new), axis=0)
            print("new RMS state size: ", self.mean.shape)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.curr_size = 0
        self.total_count = 0
        self.buffer = None


    def getRandIndex(self,n):
        idx = np.random.randint(0, self.curr_size, size=n)
        return idx

    def getSample(self, idx):
        return self.buffer[idx]

    def init_buffer(self, data):
        dtype = data[0].dtype
        shape = [self.buffer_size] + list(data[0].shape)
        self.buffer = np.zeros(shape, dtype=dtype)
        # self.RMS = RunningMeanStd(shape=list(data[0].shape))


        # if self.saved_path is not None :
        #     self.RMS.load(self.save_path)
        #     self.RMS.setNumStates(list(data[0].shape))
        return

    def clear(self):
        self.curr_size = 0
        self.total_count = 0
        self.buffer=None

    def store(self, data):
        n = len(data)

        if (n > 0):
            if self.buffer is None:
                self.init_buffer(data)
            # data_normalized = self.RMS.apply(np.array(data))
            idx = self.getStore_idx(n)
            self.buffer[idx] = data
            self.curr_size = min(self.curr_size + n, self.buffer_size)
            self.total_count += n
        return

    def get_current_size(self):
        return self.curr_size

    def is_full(self):
        return self.curr_size >= self.buffer_size

    # def saveRMS(self,path):
    #     self.RMS.save(path)
    #     return

    # def loadRMS(self,path):
    #     self.saved_path=path
    #     return

    def getStore_idx(self, n):
        assert n < self.buffer_size # bad things can happen if path is too long
        
        idx = []
        if (not self.is_full()):
            start_idx = self.curr_size
            end_idx = min(self.buffer_size, start_idx + n)
            idx = list(range(start_idx, end_idx))

        remainder = n - len(idx)
        if (remainder > 0):
            rand_idx = list(np.random.choice(self.curr_size, remainder, replace=False))
            idx += rand_idx

        return idx

        

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def get_shape_tf(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known" 
    return out

def numel_tf(x):
    out = get_shape_tf(x)
    n = int(np.prod(out))
    return n

def flatten(arr_list):
    return np.concatenate([np.reshape(a, [-1]) for a in arr_list], axis=0)

class SetFromFlat(object):
    def __init__(self, sess, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(get_shape_tf, var_list))
        total_size = np.sum([int(np.prod(shape)) for shape in shapes])

        self.sess = sess
        self.theta = tf.placeholder(dtype,[total_size])
        start=0
        assigns = []

        for (shape,v) in zip(shapes,var_list):
            size = int(np.prod(shape))
            assigns.append(tf.assign(v, tf.reshape(self.theta[start:start+size],shape)))
            start += size

        self.op = tf.group(*assigns)
        return

    def __call__(self, theta):
        self.sess.run(self.op, feed_dict={self.theta:theta})
        return

class GetFlat(object):
    def __init__(self, sess, var_list):
        self.sess = sess
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel_tf(v)]) for v in var_list])
        return

    def __call__(self):
        return self.sess.run(self.op)