# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

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

        # me aseguro de tener el parámetro obligatorio
        @assert haskey(modelHyperparameters, "topology") "Error: si vas a entrenar una RNA, asegúrate de indicar una topología"
        # obtengo el parámetro obligatorio
        topology = modelHyperparameters["topology"]

        # hago una llamada a entrenar la red neuronal con los valores del dict (si es que están sino los por defecto que pongo)
        ann = ANNCrossValidation(topology, dataset, crossValidationIndices;
        numExecutions = get(modelHyperparameters, "numExecutions", 50),
        transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
        maxEpochs = get(modelHyperparameters, "maxEpochs", 1000),
        minLoss = get(modelHyperparameters, "minLoss", 0.0),
        learningRate = get(modelHyperparameters, "learningRate", 0.01),
        validationRatio = get(modelHyperparameters, "validationRatio", 0),
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20))

        return ann

    end;

    # sino como en la práctica anterior

    inputs = dataset[1]
    targets = dataset[2]

    # paso a cadenas de texto
    targets = string.(targets)

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

        # Para cada fold i, los patrones cuyo índice es i son test y el resto train
        input_train  = inputs[crossValidationIndices .!= i, :]
        target_train = targets[crossValidationIndices .!= i, :]
        input_test   = inputs[crossValidationIndices .== i, :]
        target_test  = targets[crossValidationIndices .== i, :] 

        if modelType == :DoME 

            maximumNodes = modelHyperparameters["maximumNodes"]
            # trainingDataset = (trainInputs, vec(trainTargets))
            testOutputs = trainClassDoME((input_train, vec(target_train)), input_test, maximumNodes)

        else

            if modelType == :SVC 

                @assert haskey(modelHyperparameters, "topology") "Error: si vas a entrenar una RNA, asegúrate de indicar una topología"
                kernel = modelHyperparameters["kernel"] # obtenermos el kernel

                C = get(modelHyperparameters, "cost", 1) # este hiperparámetro es necesario en todas
                if kernel == "linear"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Linear, 
                        cost = Float64(C))
                end;
                
                gamma = get(modelHyperparameters, "gamma", 0.01)
                if kernel == "rbf"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.RadialBasis, 
                        cost = Float64(C), gamma = Float64(gamma))
                end;

                coef0 = get(modelHyperparameters, "coef0", 1)
                if kernel == "sigmoid"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Sigmoid, 
                        cost = Float64(C), gamma = Float64(gamma), coef0 = Float64(coef0))
                end;

                degree = get(modelHyperparameters, "degree", 3)
                if kernel == "poly"
                    model = SVMClassifier(kernel = LIBSVM.Kernel.Sigmoid, 
                        cost = Float64(C), gamma = Float64(gamma), coef0 = Float64(coef0), degree = Float64(degree))
                end;
            
            end;

            if modelType == :DecisionTreeClassifier
                @assert haskey(modelHyperparameters, "max_depth") "Error: Debes pasar la profundidad máxima del árbol como parámetro"
                max_depth = modelHyperparameters["max_depth"]
                model = DTClassifier(max_depth = max_depth, rng = Random.MersenneTwister(1))
            end;

            if modelType == :KNeighborsClassifier
                @assert haskey(modelHyperparameters, "n_neighbors") "Error: Debes pasar el número de vecinos que tener en cuenta"
                K = modelHyperparameters["n_neighbors"]
                model = KNNClassifier(K = K)
            end;

            mach = machine(model, MLJ.table(input_train), categorical(target_train))
            MLJ.fit!(mach, verbosity=0) 

            testOutputs = MLJ.predict(mach, MLJ.table(input_test))

            # en caso de ser DT o KNN (mirar lo del nombre de variable)
            if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
                testOutputs = mode.(testOutputs)
            end;


        end;

        acc, er, sens, spec, p, n, f1score, cm = confusionMatrix(testOutputs, vec(target_test), classes)

        accuracy[i] = acc
        errorrate[i]  = er
        sensitivity[i] = sens
        specificity[i] = spec
        ppv[i] = p
        npv[i] = n
        f1[i] = f1score

        confusionmatrix .+= cm


    end;

    return (
        (mean(accuracy), std(accuracy)),
        (mean(errorrate), std(errorrate)),
        (mean(sensitivity), std(sensitivity)),
        (mean(specificity), std(specificity)),
        (mean(ppv),  std(ppv)),
        (mean(npv), std(npv)),
        (mean(f1),  std(f1)),
        confusionmatrix)


end;
