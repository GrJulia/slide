using PyCall
using JSON
using Random
using DataSets
using Conda

Conda.add("tensorflow")

pushfirst!(PyVector(pyimport("sys")."path"), "/mnt/data/code/slide/src/tf")

train = pyimport("train")


config = JSON.parsefile("examples/configs/default_amazon.json")
config["name"] *= "_tf_" * randstring(8)
println("Name: $(config["name"])")

train_f = open(String, dataset(config["dataset"]["train_path"]))
test_f = open(String, dataset(config["dataset"]["test_path"]))

train(config, train_f, test_f)
