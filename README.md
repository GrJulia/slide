# slide
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
The `Slide` package most likely will fail to resolve. To fix that follow this steps in the `example` directory:
```
    julia
    ] # opens pkg mode
    add path_to_slide_repo#branch_name
```
Where `path_to_slide_repo` is a local path of this project and `#branch_name` is the name of the branch you are interested in.
The `#branch_name` can be ommited and by default the `main` will be used.

Note: only commited change to selected branch will be pulled as a package code.