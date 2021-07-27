import tensorflow.keras as keras
import tensorflow.nn as nn
import numpy as np
import os
import time
from collections import defaultdict
import json


class Model(keras.Model):
    def __init__(self, n_features, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.model = keras.Sequential([
            keras.layers.InputLayer(n_features),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(n_classes),
        ])

    def call(self, x):
        return self.model(x)


class SparseDataset(keras.utils.Sequence):
    def __init__(self, ds, batch_size, n_features, n_classes):
        super(SparseDataset, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.__preprocess_dataset(ds)

    def __preprocess_dataset(self, dataset):
        x_indices, x_vals, ys = [], [], []
        for line in dataset.split('\n')[1:-1]:
            line_split = line.split()
            x = [(int(ftr.split(':')[0]), float(ftr.split(':')[1])) for ftr in line_split[1:]]
            y = [int(yi) for yi in line_split[0].split(',')]
            x_indices.append(np.array([xi[0] for xi in x]))
            x_vals.append(np.array([xi[1] for xi in x]))
            ys.append(np.array(y))
    
        perm = np.random.permutation(len(ys))
        p_x_indices = [x_indices[idx] for idx in perm]
        p_x_vals = [x_vals[idx] for idx in perm]
        p_ys = [ys[idx] for idx in perm]
    
        self.xs = (p_x_indices, p_x_vals)
        self.ys = p_ys

    def __len__(self):
        return len(self.ys) // self.batch_size

    def __getitem__(self, b_idx):
        (xs_indices, xs_vals) = self.xs

        x = np.zeros((self.batch_size, self.n_features))
        y = np.zeros((self.batch_size, self.n_classes))
        for idx in range(self.batch_size):
            tr_idx = b_idx * self.batch_size + idx
            x[idx, xs_indices[tr_idx]] = xs_vals[tr_idx]
            
            ys = self.ys[tr_idx]
            y[idx, ys] = 1
        
        return x, y


class TestAccCallback(keras.callbacks.Callback):
    def __init__(self, test_dataloader, config):
        super().__init__()
        self.test_dataloader = test_dataloader
        self.freq = config["test_freq"]
        self.n_batches = config["n_batches"]
        self.use_random_indices = config["use_random_indices"]
        self.cnt = 0

    def on_train_batch_end(self, batch, logs):
        self.cnt += 1
        if self.cnt % self.freq == 0:
            logs["test_acc"] = (self.cnt, self.test_epoch())

    def test_epoch(self):
        if self.use_random_indices:
            rand_indices = np.random.randint(0, len(self.test_dataloader), self.n_batches)
        else:
            rand_indices = np.arange(self.n_batches)
        total_acc, batch_size = 0, self.test_dataloader.batch_size
        for idx in rand_indices:
            x, y = self.test_dataloader[idx]
            out = self.model(x)
            for b in range(batch_size):
                top_class = np.argmax(out[b])
                total_acc += y[b, top_class]
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
    model.compile(optimizer=keras.optimizers.Adam(config["lr"]), loss=nn.softmax_cross_entropy_with_logits, run_eagerly=True)

    train_set = SparseDataset(train_f, config["batch_size"], config["n_features"], config["n_classes"])
    test_set = SparseDataset(test_f, config["batch_size"], config["n_features"], config["n_classes"])

    log_dir = os.path.join(config["logging_path"], config["name"])

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    test_cb = TestAccCallback(test_set, config["testing"])
    time_measure_cb = TimeMeasureCallback()
    logger_cb = LoggerCallback(log_dir)
    callbacks = [tensorboard_cb, test_cb, time_measure_cb, logger_cb]

    model.fit(train_set, epochs=config["n_epochs"], callbacks=callbacks)
    