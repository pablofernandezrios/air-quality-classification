# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    cicles = collect(1:k)
    times = ceil(Int, N/k)  # Número de veces a repetir cicles
    repeatCicles = repeat(cicles, inner=times)
    repeatCicles = repeatCicles[1:N]
    return shuffle!(repeatCicles)
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Crear un vector de índices con tantos valores como filas en targets
    numRows = size(targets, 1) 
    index = collect(1:numRows)

    positiveInstances = count(targets .>0)  # Seleccionar las instancias positivas en targets
    positiveSubsets = crossvalidation(positiveInstances, k)
    index[index .> 0] .= positiveSubsets  # Asignar a las posiciones de instancias positivas el resultado de la llamada a crossValidation

    negativeInstances = count(targets .>0)  # Seleccionar las instancias negativas en targets
    negativeSubsets = crossvalidation(negativeInstances, k)
    index[index .> 0] .= negativeSubsets  # Asignar a las posiciones de instancias negativas el resultado de la llamada a crossValidation

    return index
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Crear un vector de índices con tantos valores como filas en targets
    numRows = size(targets, 1)    
    index = collect(1:numRows)

    # Para cada clase, seleccionar el número de subconjuntos correspondiente
    for indexColumn in eachindex(targets[:, 1])
        numClasses = sum(targets[indexColumn, :])
        @assert (numClasses >= k) "debe haber al menos k patrones en cada clase"
        # BAJAR EL VALOR DE K??? QUÉ CONSECUENCIAS TENDRÍA ESTO??

        subsets = crossvalidation(numClasses, k)
        index[indexColumn] = subsets
    end

    return index
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
    # Asignamos la matriz de entradas y el vector de salidas deseadas
    inputs, targets = dataset

    classes = unique(targets)
    numClasses = length(classes)  # Obtenemos el número de clases
    targets = oneHotEncoding(targets)  # Transformar el vector de salidas deseadas en una matriz de valores booleanos


    # Calcular el número de folds
    folds = maximum(crossValidationIndices)
    # Inicializar vectores para cada métrica de clasificación
    accFold = zeros(folds)
    errorRateFold = zeros(folds)
    recallFold = zeros(folds)
    specificityFold = zeros(folds)
    precisionFold = zeros(folds)
    NPVFold = zeros(folds)
    f1Fold = zeros(folds)
    # Inicializar la matriz de confusión con valores reales iguales a 0
    confusionMatrixTotal = zeros(Float64, numClasses, numClasses)

    for fold in 1:folds
        # Variables de entrada y salida deseada para entrenamiento
        trainInputs = inputs[crossValidationIndices .!= fold, :]
        trainTargets =  targets[crossValidationIndices .!= fold, :]
        # Variables de entrada y salida deseada para test
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = targets[crossValidationIndices .== fold, :]

        # Crear nuevos vectores para los resultados de cada entrenamiento para cada métrica
        accExecution = zeros(numExecutions)
        errorRateExecution = zeros(numExecutions)
        recallExecution = zeros(numExecutions)
        specificityExecution = zeros(numExecutions)
        precisionExecution = zeros(numExecutions)
        NPVExecution = zeros(numExecutions)
        f1Execution = zeros(numExecutions)
        confusionMatrixExecution = zeros(Float64, numClasses, numClasses, numExecutions)

        # Training loop
        for numExecution in 1:numExecutions

            if validationRatio > 0
                numTest = (count(crossValidationIndices .== fold))
                numTrain = (count(crossValidationIndices .!= fold))
                localValidationRatio = validationRatio / (1.0 - numTest / (numTest + numTrain))
                
                trainIdx, valIdx = holdOut(numTrain, localValidationRatio)
                validationDataset = (trainInputs[valIdx, :], trainTargets[valIdx, :])
                reducedTrainData  = (trainInputs[trainIdx, :], trainTargets[trainIdx, :])

                ann, _, _, _ = trainClassANN(topology, reducedTrainData;
                    validationDataset = validationDataset,
                    testDataset       = (testInputs, testTargets),
                    transferFunctions = transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss,
                    learningRate=learningRate, maxEpochsVal=maxEpochsVal)            
            else 
                ann, _, _, _ = trainClassANN(topology, (trainInputs, trainTargets);
                    testDataset       = (testInputs, testTargets),
                    transferFunctions = transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss,
                    learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            end
            outputs = ann(convert(Array{Float32,2}, testInputs)')'

            # Guardamos las métricas de clasificación para esta iteración
            accExecution[numExecution], 
            errorRateExecution[numExecution], 
            recallExecution[numExecution], 
            specificityExecution[numExecution], 
            precisionExecution[numExecution], 
            NPVExecution[numExecution], 
            f1Execution[numExecution], 
            confusionMatrixExecution[:, :, numExecution] = confusionMatrix(outputs, testTargets)
        end

        accFold[fold] = mean(accExecution)
        errorRateFold[fold] = mean(errorRateExecution)
        recallFold[fold] = mean(recallExecution)
        specificityFold[fold] = mean(specificityExecution)
        precisionFold[fold] = mean(precisionExecution)
        NPVFold[fold] = mean(NPVExecution)
        f1Fold[fold] = mean(f1Execution)
        
        confusionMatrixTotal .+= dropdims(mean(confusionMatrixExecution, dims=3), dims=3)
    end

    return ((mean(accFold), std(accFold)), 
            (mean(errorRateFold), std(errorRateFold)), 
            (mean(recallFold), std(recallFold)), 
            (mean(specificityFold), std(specificityFold)), 
            (mean(precisionFold), std(precisionFold)), 
            (mean(NPVFold), std(NPVFold)), 
            (mean(f1Fold), std(f1Fold)),
            confusionMatrixTotal)

end;

