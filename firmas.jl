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
    vp = count(outputs .& targets)
    vn = count(.!outputs .& .!targets)
    fp = count(outputs .& .!targets)
    fn = count(.!outputs .& targets)
    confusionmatrix = [[vn fp]; [fn vp]]

    acc = (vn + vp) / (vn + vp + fn + fp)    
    errorRate = (fn + fp) / (vn + vp + fn + fp)  # Sería equivalente a (1 - acc)
    recall = (vp + fn) == 0 ? 1.0 : (vp) / (vp + fn)
    specificity = (vn + fp) == 0 ? 1.0 : (vn) / (vn + fp)
    precision = (vp + fp) == 0 ? 1.0 : (vp) / (vp + fp)
    NPV = (vn + fn) == 0 ? 1.0 : (vn) / (vn + fn)
    f1 = (recall + precision) == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall)

    return (acc, errorRate, recall, specificity, precision, NPV, f1, confusionmatrix)
end;


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end;


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 

    metricas = confusionMatrix(outputs, targets)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])

end;
    
    
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    metricas = confusionMatrix(outputs, targets; threshold = threshold)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])

end;


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "outputs y targets deben tener el mismo tamaño"

    numClasses = size(outputs, 2)

    @assert numClasses != 2 "Para problemas binarios usa la versión unidimensional"

    if numClasses == 1  
        return confusionMatrix(vec(outputs), vec(targets))
    end

    recall = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    NPV = zeros(numClasses)
    f1 = zeros(numClasses)

    for i in 1:numClasses
        acc, error, rec, spec, prec, npv, f1score, confmatrix = confusionMatrix(outputs[:, i], targets[:, i])
        recall[i] = rec
        specificity[i] = spec
        precision[i] = prec
        NPV[i] = npv
        f1[i] = f1score
    end
    confusionmatrix = targets' * outputs

    instances = vec(sum(targets, dims=1))
    if weighted
        recall = sum(recall .* instances) / (sum(instances))
        specificity = sum(specificity .* instances) / (sum(instances))
        precision = sum(precision .* instances) / (sum(instances))
        NPV = sum(NPV .* instances) / (sum(instances))
        f1 = sum(f1 .* instances) / (sum(instances))
    else    
        recall = sum(recall) / numClasses
        specificity = sum(specificity) / numClasses
        precision = sum(precision) / numClasses
        NPV = sum(NPV) / numClasses
        f1 = sum(f1) / numClasses
    end

    # Calculamos el valor de precisión con la función accuracy desarrollada en la práctica anterior
    acc = accuracy(outputs, targets)
    errorRate = 1 - acc
    
    return (acc, errorRate, recall, specificity, 
            precision, NPV, f1, confusionmatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputs = classifyOutputs(outputs; threshold=threshold) # convertimos a matriz bool
    return (confusionMatrix(outputs, targets; weighted = weighted))
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "Ambos vectores deben tener la misma longitud"
    @assert all([in(label, classes) for label in unique(vcat(targets, outputs))]) "Las etiquetas deben estar incluidas en classes"
    
    # codificamos las matrices
    outputs = oneHotEncoding(outputs, classes)
    targets = oneHotEncoding(targets, classes)

    return (confusionMatrix(outputs, targets; weighted = weighted))
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return (confusionMatrix(outputs, targets, classes; weighted = weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    metricas = confusionMatrix(outputs, targets; weighted = weighted)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    metricas = confusionMatrix(outputs, targets; weighted = weighted)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    metricas = confusionMatrix(outputs, targets, classes; weighted = weighted)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    metricas = confusionMatrix(outputs, targets; weighted = weighted)

    println("Métricas:")
    println("Accuracy: $(metricas[1])")
    println("Error rate: $(metricas[2])")
    println("Sensitivity: $(metricas[3])")
    println("Specificity: $(metricas[4])")
    println("PPV: $(metricas[5])")
    println("NPV: $(metricas[6])")
    println("F1-score: $(metricas[7])")

    println("\nMatriz de confusión:")
    println(metricas[8])
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64, 2}, trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = convert(AbstractArray{Float64, 2}, testInputs)

    model, _, _, _ = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes) 

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

    # Bloqueamos el caso de clasificación binaria mal codificada
    @assert numclases != 2 "Error: Matriz de 2 columnas detectada. Usa un vector para problemas binarios."

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
    indices = collect(1:k) # crea identificadores de los folds para asignar cada patrón a un subconjunto
    num_repeticiones = ceil(Int, N/k) # cuántas veces repetirlos para cubrir N elementos
    indices_repetidos = repeat(indices, num_repeticiones)

    indices = indices_repetidos[1:N]
    shuffle!(indices)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = collect(1:length(targets))

    # asigna folds estratificados (positivos y negativos están distribuidos equilibradamente entre los folds)
    indices[targets .== true]  = crossvalidation(sum(targets .== true), k) 
    indices[targets .== false] = crossvalidation(sum(targets.== false), k)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = collect(1:size(targets, 1)) # vector de longitud numero de filas
    @assert all(sum(targets, dims=1) .>= k) "Error: Al menos una clase tiene menos de 10 instancias, lo cual generará una k-validación sesgada"
    for c in eachindex(1:size(targets,2))
       indices[targets[:,c] .== true] = crossvalidation(sum(targets[:,c]), k)
    end

    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)

    inputs = dataset[1]
    targetsCat = dataset[2]
    classes = unique(targetsCat)
    targets = oneHotEncoding(targetsCat, classes)

    folds = maximum(crossValidationIndices)
    numClasses = length(classes)

    accur  = zeros(folds)
    errorRate   = zeros(folds)
    sensitivity = zeros(folds)
    specificity = zeros(folds)
    ppv         = zeros(folds)
    npv         = zeros(folds)
    f1          = zeros(folds)
    confMatrixGlobal = zeros(Float64, numClasses, numClasses)

    for fold in 1:folds
        trainMask = crossValidationIndices .!= fold
        testMask  = crossValidationIndices .== fold

        inputsTrain  = inputs[trainMask, :]
        targetsTrain = targets[trainMask, :]
        inputsTest   = inputs[testMask, :]
        targetsTest  = targets[testMask, :]

        accExec  = zeros(numExecutions)
        errExec  = zeros(numExecutions)
        sensExec = zeros(numExecutions)
        specExec = zeros(numExecutions)
        ppvExec  = zeros(numExecutions)
        npvExec  = zeros(numExecutions)
        f1Exec   = zeros(numExecutions)
        confMatrices = zeros(Float64, numClasses, numClasses, numExecutions)

        for exec in 1:numExecutions
            if validationRatio > 0
                # El ratio se ajusta porque el conjunto de entrenamiento ya
                # es menor al haber separado el fold de test
                ratioVal = validationRatio / (1.0 - 1.0/folds)
                trainIdx, valIdx = holdOut(size(inputsTrain, 1), ratioVal)

                inputsTrainExec  = inputsTrain[trainIdx, :]
                targetsTrainExec = targetsTrain[trainIdx, :]
                inputsVal        = inputsTrain[valIdx, :]
                targetsVal       = targetsTrain[valIdx, :]

                ann, _ = trainClassANN(
                    topology,
                    (inputsTrainExec, targetsTrainExec);
                    transferFunctions = transferFunctions,
                    maxEpochs        = maxEpochs,
                    minLoss          = minLoss,
                    learningRate     = learningRate,
                    validationDataset = (inputsVal, targetsVal),
                    maxEpochsVal     = maxEpochsVal
                )
            else
                ann, _ = trainClassANN(
                    topology,
                    (inputsTrain, targetsTrain);
                    transferFunctions = transferFunctions,
                    maxEpochs        = maxEpochs,
                    minLoss          = minLoss,
                    learningRate     = learningRate,
                    maxEpochsVal     = maxEpochsVal
                )
            end

            outputs = ann(inputsTest')  # columnas = patrones
            acc, err, sens, spec, vpp_, vpn_, f1_, cm =
                confusionMatrix(outputs', targetsTest)

            accExec[exec]  = acc
            errExec[exec]  = err
            sensExec[exec] = sens
            specExec[exec] = spec
            ppvExec[exec]  = vpp_
            npvExec[exec]  = vpn_
            f1Exec[exec]   = f1_
            confMatrices[:, :, exec] = cm
        end

        accur[fold]    = mean(accExec)
        errorRate[fold]   = mean(errExec)
        sensitivity[fold] = mean(sensExec)
        specificity[fold] = mean(specExec)
        ppv[fold]         = mean(ppvExec)
        npv[fold]         = mean(npvExec)
        f1[fold]          = mean(f1Exec)

        meanConf = mean(confMatrices, dims=3)
        confMatrixGlobal .+= meanConf[:, :, 1]
    end

    return (
        (mean(accur),    std(accur)),
        (mean(errorRate),   std(errorRate)),
        (mean(sensitivity), std(sensitivity)),
        (mean(specificity), std(specificity)),
        (mean(ppv),         std(ppv)),
        (mean(npv),         std(npv)),
        (mean(f1),          std(f1)),
        confMatrixGlobal
    )
end


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
    if modelType == :ANN
        @assert haskey(modelHyperparameters, "topology") "Error: si vas a entrenar una RNA, asegúrate de indicar una topología"
        topology = modelHyperparameters["topology"]
        numExecutions = get(modelHyperparameters, "numExecutions", 50)
        transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
        maxEpochs = get(modelHyperparameters, "maxEpochs", 1000)
        minLoss = get(modelHyperparameters, "minLoss", 0.0)
        learningRate = get(modelHyperparameters, "learningRate", 0.01)
        validationRatio = get(modelHyperparameters, "validationRatio", 0)
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20)

        return ANNCrossValidation(topology, dataset, crossValidationIndices; numExecutions, transferFunctions, maxEpochs, minLoss, 
                    learningRate, validationRatio, maxEpochsVal)
    end

    inputs = dataset[1]
    targets = dataset[2]

    targets = string.(targets)
    classes = unique(targets)
    numfolds = maximum(crossValidationIndices)


    accur = zeros(numfolds)
    errorRate = zeros(numfolds)
    recall = zeros(numfolds)
    specificity = zeros(numfolds)
    precision = zeros(numfolds)
    npv = zeros(numfolds)
    f1 = zeros(numfolds)
    confMatrixGlobal = zeros(Float64, length(classes), length(classes))

    for fold in 1:numfolds
        trainMask = crossValidationIndices .!= fold
        testMask  = crossValidationIndices .== fold

        trainInputs  = inputs[trainMask, :]
        trainTargets = targets[trainMask, :]
        testInputs   = inputs[testMask, :]
        testTargets  = targets[testMask, :]

        if modelType == :DoME
            maximumNodes = modelHyperparameters["maximumNodes"]
            trainingDataset = (trainInputs, vec(trainTargets))
            testOutputs = trainClassDoME(trainingDataset, testInputs, maximumNodes)
        
        else
            if modelType == :SVC
                @assert haskey(modelHyperparameters, "kernel") "Error: Si quieres usar una SVM, debes indicar el kernel a utilizar!"
                kernel = modelHyperparameters["kernel"]

                C = get(modelHyperparameters, "cost", 1)
                if kernel == "linear"
                    model = SVMClassifier( 
                        kernel = LIBSVM.Kernel.Linear,  
                        cost = Float64(C))
                end
                
                gamma = get(modelHyperparameters, "gamma", 0.01)
                if kernel == "rbf"
                    model = SVMClassifier( 
                        kernel = LIBSVM.Kernel.RadialBasis,  
                        cost = Float64(C),
                        gamma = Float64(gamma))
                end
                
                coef0 = get(modelHyperparameters, "coef0", 1)
                if kernel == "sigmoid"
                    model = SVMClassifier( 
                        kernel = LIBSVM.Kernel.Sigmoid,  
                        cost = Float64(C),
                        gamma = Float64(gamma),
                        coef0 = Float64(coef0))
                end

                degree = get(modelHyperparameters, "degree", 3)
                if kernel == "poly"
                    model = SVMClassifier( 
                        kernel = LIBSVM.Kernel.Polynomial,  
                        cost = Float64(C),
                        gamma = Float64(gamma),
                        coef0 = Float64(coef0),
                        degree = Int32(degree))
                end 
            end 

            if modelType == :DecisionTreeClassifier
                @assert haskey(modelHyperparameters, "max_depth") "Error: Debes pasar la profundidad máxima del árbol como parámetro"
                max_depth = modelHyperparameters["max_depth"]
                model = DTClassifier(max_depth = max_depth, rng=Random.MersenneTwister(1)) 
            end

            if modelType == :KNeighborsClassifier
                @assert haskey(modelHyperparameters, "n_neighbors") "Error: Debes pasar el número de vecinos que tener en cuenta"
                K = modelHyperparameters["n_neighbors"]
                model = kNNClassifier(K = K) ; 
            end

            mach = machine(model, MLJ.table(trainInputs), categorical(vec(trainTargets)));
            MLJ.fit!(mach, verbosity=0)
            testOutputs = MLJ.predict(mach, MLJ.table(testInputs)); 

            if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
                testOutputs = mode.(testOutputs) 
            end
        end

        acc, err, rec, spec, prec, npv_, f1_, cm = confusionMatrix(testOutputs, vec(testTargets), classes)
        accur[fold] = acc
        errorRate[fold]  = err
        recall[fold] = rec
        specificity[fold] = spec
        precision[fold] = prec
        npv[fold] = npv_
        f1[fold] = f1_

        confMatrixGlobal .+= cm
    end 

    return (
        (mean(accur), std(accur)),
        (mean(errorRate), std(errorRate)),
        (mean(recall), std(recall)),
        (mean(specificity), std(specificity)),
        (mean(precision),  std(precision)),
        (mean(npv), std(npv)),
        (mean(f1),  std(f1)),
        confMatrixGlobal
    )

end;
