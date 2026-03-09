# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = collect(1:k) # crea identificadores de los folds para asignar cada patrón a un subconjunto
    num_repeticiones = ceil(Int, N/k) # cuántas veces repetirlos para cubrir N elementos
    indices_repetidos = repeat(indices, num_repeticiones)

    indices = indices_repetidos[1:N]
    shuffle!(indices)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = collect(1:length(targets))

    N_pos = sum(targets .== true)
    N_neg = sum(targets .== false)

    # asigna folds estratificados (positivos y negativos están distribuidos equilibradamente entre los folds)
    indices[targets .== true]  = crossvalidation(N_pos, k) 
    indices[targets .== false] = crossvalidation(N_neg, k)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = collect(1:size(targets, 1)) # vector de longitud numero de filas

    for c in eachindex(1:size(targets,2))
       indices[targets[:,c] .== true] = crossvalidation(sum(targets[:,c]), k)
    end

    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets = oneHotEncoding(targets)
    return crossvalidation(targets, k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;
