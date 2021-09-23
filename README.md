<p align="center">
    <a href="https://github.com/GrJulia/slide">
        <img src="logo.png">
    </a>
</p>

# SLiDE
Julia implementation of Sublinear Deep Learning Engine

## Installation
The following set of instructions will install all required packages from the Julia REPL.
```
julia
]
activate .
resolve
```

## Running tests

```
julia
]
activate .
test
```

## Example of usage

```
using Slide, Slide.Network, Slide.LshSimHashWrapper, Slide.Hash, Slide.Network.Layers, Slide.Network.Optimizers, Slide.Logger

input_dim, hidden_dim, output_dim = 100, 20, 500
n_samples = 4096
batch_size = 128

# Generate random data
x, y = rand(Float, input_dim, n_samples), one_hot(rand(1:output_dim, n_samples))
train_set = batch_input(x, y, batch_size, false)

# Initialize SimHash
common_lsh = LshParams(
    n_buckets = 9,
    n_tables = 50,
    max_bucket_len = 128,
)
lsh_params = get_simhash_params(
    common_lsh,
    [hidden_dim, output_dim];
    signature_len = 9,
    sample_ratio = Float(1.5),
    input_size = input_dim,
)

# Build the network
network = SlideNetwork(
    Dense(input_dim, hidden_dim, relu),
    SlideLayer(hidden_dim, output_dim, lsh_params[end], identity)
)
optimizer = AdamOptimizer(eta = 0.0001),

# Initialize the logger
logger = get_logger(Dict("logging_path" => "./logs", "use_tensorboard" => false), "toy_example")
global_logger(logger)

# Train the network
train!(
    train_set,
    network,
    logger;
    n_epochs = 10,
)

```

## Reproducing SLIDE
In order to train SLIDE on the real data, adjust paths in examples/configs/default_delicious.json and do following:
'''
julia -t <n_threads>
]
activate .
include("examples/main.jl")
'''


## Running examples
Go into the `examples` directory and run example with julia command with added project argument like so:
```bash
    julia --project=example file_to_run.jl ...
```
The `Slide` package will most likely fail to resolve. To fix this, follow these steps in the `example` directory:
```
    julia
    ] # opens pkg mode
    add <path_to_slide_repo><#branch_name>
```
Where `path_to_slide_repo` is a local path of this project and `#branch_name` is the name of the branch you are interested in.
The `#branch_name` can be ommitted and by default the `main` branch will be used.

Note: only committed changes to the selected branch will be pulled as package code.
