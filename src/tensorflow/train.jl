using PyCall
using JSON
using Random

include("data.jl")

# os = pyimport("os")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        for cnt, line in enumerate(f.readlines()[1:]):
            line_split = line.split()
            x = list(map(lambda ftr: (int(ftr.split(':')[0]), float(ftr.split(':')[1])), line_split[1:]))
            y = [int(yi) for yi in line_split[0].split(',')]
            x_indices.append(np.array([xi[0] for xi in x]))
            x_vals.append(np.array([xi[1] for xi in x]))
            ys.append(np.array(y))
            # if cnt == 2000:
            #     break
    
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

class TestAccCallback(keras.callbacks.Callback):
    def __init__(self, test_dataloader, freq, n_batches):
        super().__init__()
        self.test_dataloader = test_dataloader
        self.freq = freq
        self.n_batches = n_batches
        self.cnt = 0

    def on_train_batch_end(self, i, batch_logs):
        self.cnt += 1
        if self.cnt % self.freq == 0:
            self.cnt = 0
            self.test_epoch()

    def test_epoch(self):
        rand_indices = np.random.randint(0, self.test_dataloader.__len__(), self.n_batches)
        batch_size = self.test_dataloader.batch_size
        acc = 0
        for idx in rand_indices:
            x, y = self.test_dataloader.__getitem__(idx)
            out = self.model(x)
            for b in range(batch_size):
                top_class = np.argmax(out[b])
                acc += y[b, top_class]
        print("Accuracy=", acc/(self.n_batches*batch_size)*100)

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


# @pydef mutable struct TestAccCallback <: keras.callbacks.Callback
#     function __init__(self, test_dataloader, freq, n_batches)
#         __init__(self) = pybuiltin(:super)()[:__init__]()
#         self.test_dataloader = test_dataloader
#         self.freq = freq
#         self.n_batches = n_batches
#         self.cnt = 0
#     end

#     function on_train_batch_end(self, i, batch_logs)
#         self.cnt += 1
#         if self.cnt % self.freq == 0
#             self.cnt = 0
#             self.test_epoch()
#         end
#     end

#     function test_epoch(self)
#         rand_indices = randperm(self.n_batches)
#         acc = 0
#         for idx in rand_indices
#             x, y = self.test_dataloader.__getitem__(idx)
#             println(size(x), size(y))
#             out = self.model(x)
#             println(out[1:5])
#             for b in 1:128
#                 top_k_classes = argmax(out[b, :])
#                 acc += y[b, top_k_classes]
#             end
#         end
#         println("Accuracy=$(acc/self.n_batches*128)")
#     end
# end

model = py"Model"(config["n_features"], config["hidden_dim"], config["n_classes"])

model.compile(optimizer=keras.optimizers.Adam(config["lr"]), loss=keras.losses.CategoricalCrossentropy(), run_eagerly=true)

trainset = py"SparseDataset"(config["dataset"]["train_path"], 128, config["n_features"], config["n_classes"])
testset = py"SparseDataset"(config["dataset"]["test_path"], 128, config["n_features"], config["n_classes"])

test_cb = py"TestAccCallback"(testset, 50, 40)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=joinpath(config["logging_path"], config["name"]))

model.fit(trainset, epochs=config["n_epochs"], callbacks=[test_cb, tensorboard_cb])
