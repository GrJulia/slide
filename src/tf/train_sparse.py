import tensorflow.keras as keras
import tensorflow.nn as nn
import tensorflow as tf
import numpy as np
import os
import time
from collections import defaultdict
import json
import math
import functools
from typing import Tuple, List


class SparseDense(keras.layers.Layer):
    def __init__(self, n_features_in, n_features_out):
        super(SparseDense, self).__init__()
        glorot_stddev = math.sqrt(2.0/(n_features_in+n_features_out))
        self.W = tf.Variable(tf.random.normal([n_features_in, n_features_out], stddev=glorot_stddev))
        self.B = tf.Variable(tf.random.normal([n_features_out], stddev=glorot_stddev))

    def call(self, x):
        return nn.relu(tf.sparse.sparse_dense_matmul(x, self.W) + self.B)


class Model(keras.Model):
    def __init__(self, n_features, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.model = keras.Sequential([
            keras.layers.InputLayer(n_features, sparse=True),
            SparseDense(n_features, hidden_dim),
            keras.layers.Dense(n_classes, kernel_initializer="glorot_normal", bias_initializer="glorot_normal"),
            keras.layers.Softmax(),
        ])

    def call(self, x):
        return self.model(x)


def preprocess_dataset(dataset_path, batch_size, n_samples) -> Tuple[List[np.ndarray], List[List[float]], List[np.ndarray]]:
    x_indices, x_vals, raw_ys = [], [], []
    cnt = 0
    for line in dataset.split('\n')[1:-1]:
        line_split = line.split()
        x = [(int(ftr.split(':')[0]), float(ftr.split(':')[1])) for ftr in line_split[1:]]
        y = [int(yi) for yi in line_split[0].split(',')]
        x_indices.append(np.array([xi[0] for xi in x]))
        x_vals.append([xi[1] for xi in x])
        raw_ys.append(np.array(y))
        cnt += 1
        if n_samples is not None and cnt == batch_size * n_samples:
            break

    perm = np.random.permutation(len(raw_ys))
    xs_indices = [x_indices[idx] for idx in perm]
    xs_vals = [x_vals[idx] for idx in perm]
    ys = [raw_ys[idx] for idx in perm]

    return xs_indices, xs_vals, ys


def gen_sparse_dataset(dataset, batch_size, n_features, n_classes, n_samples=None):
    xs_indices, xs_vals, ys = preprocess_dataset(dataset, batch_size, n_samples)

    for batch_idx in range(len(ys)//batch_size):
        x_indices, x_vals = [], []
        y = np.zeros((batch_size, n_classes))
        for i in range(batch_size):
            true_idx = batch_idx * batch_size + i

            x_indices += [[i, x_index] for x_index in xs_indices[true_idx]]
            x_vals += xs_vals[true_idx]

            y[i, ys[true_idx]] = 1

        yield (np.array(x_indices), np.array(x_vals), np.array([batch_size, n_features]), y)


class TestAccCallback(keras.callbacks.Callback):
    def __init__(self, test_set, config, batch_size):
        super().__init__()
        self.test_set = test_set
        self.freq = config["test_freq"]
        self.n_batches = config["n_batches"]
        self.batch_size = batch_size
        self.cnt = 0

    def on_train_batch_end(self, batch, logs):
        self.cnt += 1
        if self.cnt % self.freq == 0:
            logs["test_acc"] = (self.cnt, self.test_epoch())

    def test_epoch(self):
        n_true_positives, cnt, batch_size = 0, 0, self.batch_size
        for x, y in self.test_set:
            out = self.model(x)
            top_class = np.argmax(out, axis=1).reshape(batch_size, 1)
            n_true_positives += np.sum(np.take_along_axis(np.array(y), top_class, axis=1))
            cnt += 1
        assert(cnt == self.n_batches)
        test_acc = total_acc / (self.n_batches * batch_size)
        return test_acc


class TimeMeasureCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.cnt = 0

    def on_train_batch_begin(self, batch, logs):
        self.t0 = time.perf_counter()            

    def on_train_batch_end(self, batch, logs):
        logs["train_step time"] = (self.cnt, time.perf_counter() - self.t0)


class LoggerCallback(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_path = os.path.join(log_dir, "logs.json")
        self.cnt = 0
        self.logs = defaultdict(list)       

    def on_train_batch_end(self, batch, batch_logs):
        self.cnt += 1
        for key, val in batch_logs.items():
            if isinstance(val, tuple):
                self.logs[key].append(val)
            else:
                self.logs[key].append((self.cnt, val))

        if self.cnt % 20 == 0:
            self.save()

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)


def train(config, train_f, test_f):
    model = Model(config["n_features"], config["hidden_dim"], config["n_classes"])
    model.compile(optimizer=keras.optimizers.Adam(config["lr"]), loss=keras.losses.CategoricalCrossentropy(), run_eagerly=True)

    train_gen_with_args = functools.partial(gen_sparse_dataset, train_f, config["batch_size"], config["n_features"], config["n_classes"])
    test_gen_with_args = functools.partial(gen_sparse_dataset, test_f, config["batch_size"], config["n_features"], config["n_classes"], n_samples=config["testing"]["n_batches"])

    train_set = (tf.data.Dataset.from_generator(train_gen_with_args, (tf.int64, tf.float32, tf.int64, tf.float32))
                .map(lambda i, v, s, y: (tf.SparseTensor(i, v, s), y)))
    test_set = (tf.data.Dataset.from_generator(test_gen_with_args, (tf.int64, tf.float32, tf.int64, tf.float32))
                .map(lambda i, v, s, y: (tf.SparseTensor(i, v, s), y)))

    log_dir = os.path.join(config["logger"]["logging_path"], config["name"])

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    test_cb = TestAccCallback(test_set, config["testing"], config["batch_size"])
    time_measure_cb = TimeMeasureCallback()
    logger_cb = LoggerCallback(log_dir)
    callbacks = [tensorboard_cb, time_measure_cb, test_cb, logger_cb]

    model.fit(train_set, epochs=config["n_epochs"], callbacks=callbacks)
    