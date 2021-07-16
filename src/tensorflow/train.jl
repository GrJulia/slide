using PyCall
using JSON

include("data.jl")

keras = pyimport("tensorflow.keras")


config = JSON.parsefile(ARGS[1])

py"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class Model(keras.Model):
    def __init__(self, n_features, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.model = keras.Sequential([
            keras.layers.InputLayer(n_features),
            keras.layers.Dense(hidden_dim),
            keras.layers.Dense(n_classes),
            keras.layers.Softmax()
        ])

    def call(self, x):
        return self.model(x)

class SparseDataset(keras.utils.Sequence):
    def __init__(self, ds_path, batch_size, n_features, n_classes):
        super(SparseDataset, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.__preprocess_dataset(ds_path)

    def __preprocess_dataset(self, dataset_path):
        f = open(dataset_path, "r")
        x_indices, x_vals, ys = [], [], []
        for line in f.readlines()[1:]:
            line_split = line.split()
            x = list(map(lambda ftr: (int(ftr.split(':')[0]), float(ftr.split(':')[1])), line_split[1:]))
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
        return int(len(self.ys) / self.batch_size)

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

# def train():
#     model = Model(config["n_features"], config["hidden_dim"], config["n_classes"])
#     model.compile(optimizer="Adam", loss="mse")
#     ds = SparseDataset(config["dataset"]["train_path"], 128, config["n_features"], config["n_classes"])
#     model.fit(ds, epochs=config["n_epochs"], verbose=2)
"""

# @pydef mutable struct Model <: keras.Model
#     function __init__(self, n_features, hidden_dim, n_classes)
#         __init__(self) = pybuiltin(:super)(Model, self)[:__init__]()
#         self.model =  keras.Sequential([
#             keras.layers.Input(n_features),
#             keras.layers.Dense(hidden_dim),
#             keras.layers.Dense(n_classes),
#             keras.layers.Softmax()
#         ])
#     end

#     function call(self, x)
#         println(size(x))
#         return self.model(x)
#     end
# end

# model = keras.Sequential([
#     keras.layers.InputLayer(config["n_features"]),
#     keras.layers.Dense(config["hidden_dim"]),
#     keras.layers.Dense(config["n_classes"]),
#     keras.layers.Softmax()
# ])

model = py"Model"(config["n_features"], config["hidden_dim"], config["n_classes"])

model.compile(optimizer=keras.optimizers.Adam(config["lr"]), loss=keras.losses.CategoricalCrossentropy(), run_eagerly=true)

ds = py"SparseDataset"(config["dataset"]["train_path"], 128, config["n_features"], config["n_classes"])

model.fit(ds, epochs=config["n_epochs"])
