<p align="center">
    <a href="https://github.com/GrJulia/slide">
        <img src="logo.png">
    </a>
</p>

# SLiDE
Julia implementation of Sublinear Deep Learning Engine

## Running tests

```
julia # opens julia repl
]     # opens pkg mode
activate .
test
```

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
