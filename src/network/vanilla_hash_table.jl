const Id = Int

mutable struct HashTable
    buckets::Vector{Vector{Id}}
end

function get_random_hash(hash_table, x)
    return rand(1:length(hash_table.buckets))
end

function get_deterministic_hash(hash_table, x)
    return floor(Int, sum(x)) % length(hash_table.buckets) + 1
end

function retrieve_ids_from_bucket(hash_table, input_hash)
    return hash_table.buckets[input_hash]
end

function store_neurons_in_bucket(hash_table, neurons::Vector)
    for neuron in neurons
        neurons_hash = get_random_hash(hash_table, neuron)
        push!(hash_table.buckets[neurons_hash], neuron.id)
    end
end
