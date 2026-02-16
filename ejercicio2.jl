# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses
using Revise # Se puede eliminar después, es para ir actualizando el REPL mientras se prueba


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes)<=2 
        encodedfeatures = reshape(feature.==classes[1], :, 1);

    else  
        # Si hay más de 2 clases, se inicializa un vector de booleanos y se
        # itera sobre las categorías codificando según la clase que corresponda
        encodedfeatures = BitArray{2}(undef, length(feature), length(classes))
        for i in 1:length(classes)
            encodedfeatures[:, i] = feature .== classes[i]
        end
    end

    return encodedfeatures;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = (classes = unique(feature); oneHotEncoding(feature, classes))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minvector = minimum(dataset, dims=1);
    maxvector = maximum(dataset, dims=1);

    return (minvector, maxvector)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_vector = mean(dataset, dims = 1);
    std_vector = std(dataset, dims = 1);

    return (mean_vector, std_vector)
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset.-= minValues;
    dataset./= (maxValues .- minValues);
    dataset[:, vec(minValues .== maxValues)] .= 0;
    return dataset
end;

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) =
    (normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    normalizeMinMax!(dataset, normalizationParameters))


function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    datasetCopy = copy(dataset);
    return normalizeMinMax!(datasetCopy, normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    datasetCopy = copy(dataset);
    return normalizeMinMax!(datasetCopy);
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanvector, stdvector = normalizationParameters

    dataset .-= meanvector
    dataset ./= stdvector
    # Si la desviación estándar es 0, dividir por 0 dará error, lo corregimos
    dataset[:, vec(stdvector .== 0)] .= 0

    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationparameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationparameters)

    return dataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    datasetcopy = copy(dataset)
    return normalizeZeroMean!(datasetcopy, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    datasetcopy = copy(dataset)
    return normalizeZeroMean!(datasetcopy)
end;

classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) = outputs .>= threshold

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        return reshape(classifyOutputs(vec(outputs); threshold), :, 1)
    
    # Si hay 2 columnas, se ha representado un problema de clasificación binaria 
    # erróneamente, pero el else funcionará bien, salvo que ahora busca el máximo en vez de el mayor de 0.5
    else 
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2); 
        result = falses(size(outputs));
        result[indicesMaxEachInstance] .= true
        return result
    end 

end;

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets)

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(outputs) == size(targets) "Error: Las matrices de entrada tienen dimensiones distintas"
    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    # Si hay 2 columnas, la clasificación binaria se ha representado de manera redundante, pero
    # el else funcionará igual de bien para calcular la precisión.
    else
        return mean(eachrow(outputs) .== eachrow(targets))
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    classified = classifyOutputs(outputs; threshold = threshold)
    return accuracy(classified, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold = threshold)
    else
        classified = classifyOutputs(outputs; threshold = threshold)
        return accuracy(classified, targets)
    end
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()
    numInputsLayer = numInputs  

    # Indexamos con enumerate para no crear una variable i a mayores
    for (i, numOutputsLayer) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i])); 
        numInputsLayer = numOutputsLayer; 
    end; 

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end;



function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    features = convert(Array{Float32, 2}, dataset[1])' # Trasponemos antes de nada
    labels = convert(Array{Float32, 2}, dataset[2])'
    @assert (size(features, 2)==size(labels, 2)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de patrones"
    numInputs = size(features, 1) # Ahora están en las filas
    numOutputs = size(labels, 1)

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions)
    opt_state = Flux.setup(Adam(learningRate), ann)
    currentloss = convert(Float32, loss(ann, features, labels))
    losses = Float32[(currentloss)] # Si nunca se llamase a flux.train el vector de pérdidas tendría un elemento

    for epoch in 1:maxEpochs
        if currentloss <= minLoss  
            break
        end 
        Flux.train!(loss, ann, [(features, labels)], opt_state)
        currentloss = convert(Float32, loss(ann, features, labels))
        push!(losses, currentloss)
    end 

    return (ann, losses)

end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    return trainClassANN(topology, (inputs, reshape(targets, (:, 1))); transferFunctions, maxEpochs, minLoss, learningRate)
end;
