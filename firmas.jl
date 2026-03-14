# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses
using Revise

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
    labels = dataset[2]'
    @assert (size(features, 2)==size(labels, 2)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de patrones"
    numInputs = size(features, 1) # Ahora están en las filas
    numOutputs = size(labels, 1)

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions)
    opt_state = Flux.setup(Adam(learningRate), ann)
    currentloss = convert(Float32, loss(ann, features, labels))
    losses = Float32[currentloss] # Si nunca se llamase a flux.train el vector de pérdidas tendría un elemento

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


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    randomized = randperm(N)
    testidx = randomized[1:round(Int, (N*P))] 
    trainidx = randomized[(round(Int, N*P) + 1):end]
    return (trainidx, testidx)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    remaining, testidx = holdOut(N, Ptest)
    trainidx, validx = holdOut(length(remaining), Pval*(N / length(remaining)))

    return (remaining[trainidx], remaining[validx], testidx)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    onlytraining = size(validationDataset[1], 1) > 0 ? false : true

    trainingfeatures = convert(Array{Float32, 2}, trainingDataset[1])' 
    traininglabels = trainingDataset[2]'
    @assert (size(trainingfeatures, 2)==size(traininglabels, 2)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de patrones"
    numInputs = size(trainingfeatures, 1) # Ahora los atributos están en filas
    numOutputs = size(traininglabels, 1)  # Y las clases en filas

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions)
    opt_state = Flux.setup(Adam(learningRate), ann)
    trainingloss = convert(Float32, loss(ann, trainingfeatures, traininglabels))

    traininglosses = Float32[trainingloss]
    testlosses = Float32[]
    validationlosses = Float32[]

    if !onlytraining
        validationfeatures = convert(Array{Float32, 2}, validationDataset[1])' 
        validationlabels = validationDataset[2]'
        # Calculamos el error en el conjunto de validación
        validationloss = convert(Float32, loss(ann, validationfeatures, validationlabels))
        bestloss = deepcopy(validationloss)  # Variable que guardará la menor pérdida obtenida en validación hasta el momento
        push!(validationlosses, validationloss)

        testfeatures = convert(Array{Float32, 2}, testDataset[1])' 
        testlabels = testDataset[2]'
        # Calculamos el error en el conjunto de test
        testloss = convert(Float32, loss(ann, testfeatures, testlabels))
        push!(testlosses, testloss)

        bestann = deepcopy(ann)
    end

    worseEpochs = 0  # Contador de epochs sin mejorar pérdida con respecto a validación
    for epoch in 1:maxEpochs
        if (trainingloss <= minLoss) || (!onlytraining && (worseEpochs >= maxEpochsVal)) 
            break
        end

        Flux.train!(loss, ann, [(trainingfeatures, traininglabels)], opt_state)
        trainingloss = convert(Float32, loss(ann, trainingfeatures, traininglabels))
        push!(traininglosses, trainingloss)

        if !onlytraining
            testloss = convert(Float32, loss(ann, testfeatures, testlabels))
            validationloss = convert(Float32, loss(ann, validationfeatures, validationlabels))
            push!(testlosses, testloss)
            push!(validationlosses, validationloss)

            if validationloss > bestloss
                worseEpochs += 1
            else
                worseEpochs = 0
                bestann = deepcopy(ann)
                bestloss = deepcopy(validationloss)
            end
        end
    end

    ann = onlytraining ? ann : bestann
    return (ann, traininglosses, validationlosses, testlosses)

end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], (:, 1)))
    validationDataset = (validationDataset[1], reshape(validationDataset[2], (:, 1)))
    testDataset = (testDataset[1], reshape(testDataset[2], (:, 1)))
    
    return trainClassANN(topology, trainingDataset; validationDataset, testDataset, transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert length(outputs) == length(targets) "Los vectores outputs y targets deben tener la misma longitud"

    # calculamos los valores de la matriz de confusión
    VP = sum(outputs .& targets)
    VN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # calculamos las métricas
    accuracy = (VN + VP) / (VN + VP + FN + FP)
    errorrate = (FN + FP) / (VN + VP + FN + FP)
    sensitivity = (VP + FN) == 0 ? 1.0 : VP / (VP + FN)
    specificity = (VN + FP) == 0 ? 1.0 : VN / (VN + FP)
    ppv = (VP + FP) == 0 ? 1.0 : VP / (VP + FP)
    npv = (VN + FN) == 0 ? 1.0 : VN / (VN + FN)
    f1 = (ppv + sensitivity) == 0 ? 0.0 :
        2 * (ppv * sensitivity) / (ppv + sensitivity)

    confusionmatrix = [VN FP;
                        FN VP]

    return (accuracy = accuracy, errorrate = errorrate, sensitivity = sensitivity, specificity = specificity, 
    ppv = ppv, npv = npv, f1 = f1, confusionmatrix = confusionmatrix) # devuelve namedtuple para acceder más facilmente

end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    @assert length(outputs) == length(targets) "Los vectores outputs y targets deben tener la misma longitud"

    outputs_bool = outputs .>= threshold  # comprueba si cada uno de los valores del output es mayor que el umbral (True)
    return confusionMatrix(outputs_bool, targets)

end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 

    metricas = confusionMatrix(outputs, targets)

    println("Métricas:")
    println("Accuracy: ", metricas.accuracy)
    println("Error rate: ", metricas.errorrate)
    println("Sensitivity: ", metricas.sensitivity)
    println("Specificity: ", metricas.specificity)
    println("PPV: ", metricas.ppv)
    println("NPV: ", metricas.npv)
    println("F1-score: ", metricas.f1)

    println("\nMatriz de confusión:")
    println(metricas.confusionmatrix)
end;
    
    
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    metricas = confusionMatrix(outputs, targets, threshold = threshold)

    println("Métricas:")
    println("Accuracy: ", metricas.accuracy)
    println("Error rate: ", metricas.errorrate)
    println("Sensitivity: ", metricas.sensitivity)
    println("Specificity: ", metricas.specificity)
    println("PPV: ", metricas.ppv)
    println("NPV: ", metricas.npv)
    println("F1-score: ", metricas.f1)

    println("\nMatriz de confusión:")
    println(metricas.confusionmatrix)

end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    @assert size(outputs,2) == size(targets,2) "El número de columnas de outputs y targets debe ser igual"
    @assert size(outputs,2) != 2 "Esta función es solo para multiclase (más de 2 clases)"

    if size(outputs,2) == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end

    numclases = size(outputs, 2)

    # reservamos memoria para las métricas (1 posición para cada clase)
    sensitivity = zeros(numclases)
    specificity = zeros(numclases)
    ppv = zeros(numclases)
    npv = zeros(numclases)
    f1 = zeros(numclases)

    for c in 1:numclases
        # cogemos todas las filas de la columna c
        outputscol = outputs[:, c] 
        targetscol = targets[:, c]
        # calculamos las metricas
        metricas = confusionMatrix(outputscol, targetscol)
        # guardamos los resultados 
        sensitivity[c] = metricas.sensitivity
        specificity[c] = metricas.specificity
        ppv[c] = metricas.ppv
        npv[c] = metricas.npv
        f1[c] = metricas.f1
    end

    confusionmatrix = targets' * outputs # filas * columnas (N x C)' * (N x C)

    # peso de cada clase = nº de ejemplos de esa clase

    instancias = vec(sum(targets, dims=1)) # sumar por dims=1 suma todas las filas

    if weighted
        sensitivity = sum(sensitivity .* instancias) / sum(instancias)
        specificity = sum(specificity .* instancias) / sum(instancias)
        ppv = sum(ppv .* instancias) / sum(instancias)
        npv = sum(npv .* instancias) / sum(instancias)
        f1 = sum(f1 .* instancias) / sum(instancias)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        ppv = mean(ppv)
        npv = mean(npv)
        f1 = mean(f1)
    end

    # calculamos la accuracy con la función de una práctica anterior y la tasa de error
    accuracy = accuracy(outputs, targets)
    errorrate = 1 - accuracy

    return (accuracy = accuracy, errorrate = errorrate, sensitivity = sensitivity, specificity = specificity, 
    ppv = ppv, npv = npv, f1 = f1, confusionmatrix = confusionmatrix)
+
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputs = classifyOutputs(outputs, threshold=threshold) # convertimos a matriz bool
    return (confusionMatrix(outputs, targets, weighted))
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "Ambos vectores deben tener la misma longitud"
    @assert all([in(label, classes) for label in unique(vcat(targets, outputs))]) "Las etiquetas deben estar incluidas en classes"
    
    # codificamos las matrices
    outputs = oneHotEncoding(outputs, classes)
    targets = oneHotEncoding(targets, classes)

    return (confusionMatrix(outputs, targets, weighted))
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return (confusionMatrix(outputs, targets, classes, weighted = weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    println(confusionMatrix(outputs, targets, classes; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;


using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64, 2}, trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = convert(AbstractArray{Float64, 2}, testInputs)

    model, _, _, _ = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes) 
    model = string(model)

    testOutputs = evaluateTree(model, testInputs)
    if isa(testOutputs, Real) 
        testOutputs = repeat([testOutputs], size(testInputs,1))
    end

    return testOutputs
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    numclases = size(trainingTargets, 2)

    # en caso de solo haber una clase
    if numclases == 1
        trainingTargets = vec(trainingDataset[2])
        resultado = trainClassDoME((trainingInputs, trainingTargets), testInputs, maximumNodes)
        return reshape(resultado, :, 1)
    end

    # creamos matriz para salidas con tantas filas como instancias y columnas como clases
    numinstancias = size(testInputs, 1)
    matrizsalidas = Array{Float64,2}(undef, numinstancias, numclases)

    for c in 1:numclases
        resultados = trainClassDoME((trainingInputs, trainingTargets[:,c]), testInputs, maximumNodes)
        matrizsalidas[:,c] = resultados
    end

    return matrizsalidas

end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    classes = unique(trainingTargets)

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs,1))
    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes)
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]
        if length(classes)==2 
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else
        for numclase in eachindex(classes)
            testOutputs[testOutputsBool[:,numclase]] .= classes[numclase]; 
        end
    end

    return testOutputs

end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
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


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;


