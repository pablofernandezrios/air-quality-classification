# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes)<=2
        feature = reshape(feature.==classes[1], :, 1);
    else
        oneHot = convert(BitArray{2}, hcat([instance.==classes for instance in feature]...)');
        feature = oneHot;
    end;
    return feature;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = (classes = unique(feature); oneHotEncoding(feature, classes))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1);
    maxs = maximum(dataset, dims=1);

    return reshape([mins; maxs], 1, :);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_zero = mean(dataset, dims = 1);
    std_one = std(dataset, dims = 1);

    return reshape([mean_zero;std_one], 1, :);
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset.-= minValues;
    dataset./= (maxValues .- minValues);
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    datasetCopy = copy(dataset);
    return normalizeMinMax!(datasetCopy, normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    datasetCopy = copy(dataset);
    return normalizeMinMax!(datasetCopy);
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #
    # Codigo a desarrollar
    #
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #
    # Codigo a desarrollar
    #
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;
