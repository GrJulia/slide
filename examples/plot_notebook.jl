### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ edd732b7-eea8-4caf-a781-64421b25246d
begin
    cd("/mnt/data/code/slide")
    using Pkg
    Pkg.activate(".")
    using JSON
    using Plots
    plotly()
end

# ╔═╡ da3cb23d-ac31-44f8-823e-4d2a2ed818a5
log_path = "./examples/logs/default_delicious/logs.json"

# ╔═╡ bed2d255-8fdc-4eb5-b5aa-7fe21f9fa42f
begin
    logs_raw = JSON.parsefile(log_path)
    logs_dict = JSON.parse(logs_raw)
end

# ╔═╡ 2ba3d142-b268-41d2-8760-773a745462e0
begin
    steps_accuracy = [x[1] for x in logs_dict["test_acc"]]
    test_accuracy = [x[2] for x in logs_dict["test_acc"]]
    steps_loss = [x[1] for x in logs_dict["train_loss"]]
    train_loss = [x[2] for x in logs_dict["train_loss"]]
    steps_timings = [x[1] for x in logs_dict["train_step time"]]
    timings = [x[2] for x in logs_dict["train_step time"]]
    cumulated_timings = cumsum(timings)
end

# ╔═╡ 723a5cbf-6a12-4a38-8b41-320afb3ad564
plot(steps_accuracy, test_accuracy)

# ╔═╡ 421d1325-4a87-44a9-ad16-4d8bef408c59
plot(steps_loss, train_loss)

# ╔═╡ 8be25fad-0583-4c54-9dd3-1e935bfbaec5
plot(steps_timings, timings)

# ╔═╡ 2ffd071c-1f6e-48a9-9a25-eff2f8e4bdc6
plot(steps_timings, cumulated_timings)

# ╔═╡ Cell order:
# ╠═edd732b7-eea8-4caf-a781-64421b25246d
# ╠═da3cb23d-ac31-44f8-823e-4d2a2ed818a5
# ╠═bed2d255-8fdc-4eb5-b5aa-7fe21f9fa42f
# ╠═2ba3d142-b268-41d2-8760-773a745462e0
# ╠═723a5cbf-6a12-4a38-8b41-320afb3ad564
# ╠═421d1325-4a87-44a9-ad16-4d8bef408c59
# ╠═8be25fad-0583-4c54-9dd3-1e935bfbaec5
# ╠═2ffd071c-1f6e-48a9-9a25-eff2f8e4bdc6
