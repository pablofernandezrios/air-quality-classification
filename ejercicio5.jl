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
    
    inputs = dataset[1]
    targets = dataset[2]

    classes = unique(targets) # calcula el vector de clases
    targets = oneHotEncoding(targets, classes)

    num_folds = maximum(crossValidationIndices)
    num_classes = length(classes)

    # creamos un vector para cada métrica
    accuracy = zeros(num_folds)
    errorrate = zeros(num_folds)
    sensitivity = zeros(num_folds)
    specificity = zeros(num_folds)
    ppv = zeros(num_folds)
    npv = zeros(num_folds)
    f1 = zeros(num_folds)

    confusionmatrix = zeros(Int, num_classes, num_classes)

    for i in 1:num_folds

        # trainMask = crossValidationIndices .!= fold
        # testMask  = crossValidationIndices .== fold

        # Para cada fold i, los patrones cuyo índice es i son test y el resto train
        input_train  = inputs[crossValidationIndices .!= i, :]
        target_train = targets[crossValidationIndices .!= i, :]
        input_test   = inputs[crossValidationIndices .== i, :]
        target_test  = targets[crossValidationIndices .== i, :]

        accuracyexec = zeros(numExecutions)
        errorrateexec = zeros(numExecutions)
        sensitivityexec = zeros(numExecutions)
        specificityexec = zeros(numExecutions)
        ppvexec = zeros(numExecutions)
        npvexec = zeros(numExecutions)
        f1exec = zeros(numExecutions)
        confusionmatrixexec = zeros(Int, num_classes, num_classes, numExecutions)

        for j in 1:numExecutions

            if validationRatio > 0
            # validationRatio es una fracción del total, pero input_train es más pequeño
            # porque ya se separaron datos de test. Se escala el ratio para mantener
            # el mismo número absoluto de patrones de validación que sobre el total.
            ratio_ajustado = validationRatio * size(inputs, 1) / size(input_train, 1)
            
            # holdOut devuelve índices de entrenamiento y validación
            train_idx, val_idx = holdOut(size(input_train, 1), ratio_ajustado)
            
            # Entrenar usando el subconjunto de validación
            ann, _ = trainClassANN(topology, 
                (input_train[train_idx, :], target_train[train_idx, :]),
                validationDataset = (input_train[val_idx, :], target_train[val_idx, :]);
                transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal)
            else
            
            # Sin validación: entrenar directamente con todos los patrones del fold
                ann, _ = trainClassANN(topology,
                    (input_train, target_train);
                    transferFunctions, maxEpochs, minLoss, learningRate)
            
            end;

            outputs_test = ann(input_test') # cada valor es la activación de esa neurona de salida para ese patrón

            acc, err, sens, spec, p, n, f, cm = confusionMatrix(outputs_test, target_test)
    
            # Guardar cada métrica en la posición j del vector (una por ejecución)
            accuracyexec[j]    = acc
            errorrateexec[j]   = err
            sensitivityexec[j] = sens
            specificityexec[j] = spec
            ppvexec[j]         = p
            npvexec[j]         = n
            f1exec[j]          = f
            # Guardar la matriz de confusión en la capa j del array 3D
            confusionmatrixexec[:, :, j] = cm

        end;

        accuracy[i]    = mean(accuracyexec)
        errorrate[i]   = mean(errorrateexec)
        sensitivity[i] = mean(sensitivityexec)
        specificity[i] = mean(specificityexec)
        ppv[i]         = mean(ppvexec)
        npv[i]         = mean(npvexec)
        f1[i]          = mean(f1exec)

        confusionmatrixmean = mean(confusionmatrixexec, dims = 3)
        confusionmatrix = confusionmatrix + confusionmatrixmean[:,:,1] # tengo que extraer las dos dimensiones antes de sumar

    end;

return (
        (mean(accuracy),     std(accuracy)),
        (mean(errorrate),    std(errorrate)),
        (mean(sensitivity),  std(sensitivity)),
        (mean(specificity),  std(specificity)),
        (mean(ppv),          std(ppv)),
        (mean(npv),          std(npv)),
        (mean(f1),           std(f1)),
        confusionmatrix
    )

end;