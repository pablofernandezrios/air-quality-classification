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
        # topology es obligatorio
        topology = modelHyperparameters["topology"]
        
        # Devolvemos el resultado de llamar a ANNCrossValidation con las claves presentes en el diccionario
        return ANNCrossValidation(
            topology, dataset, crossValidationIndices;
            numExecutions     = get(modelHyperparameters, "numExecutions",     50),
            transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
            maxEpochs         = get(modelHyperparameters, "maxEpochs",         1000),
            minLoss           = get(modelHyperparameters, "minLoss",           0.0),
            learningRate      = get(modelHyperparameters, "learningRate",      0.01),
            validationRatio   = get(modelHyperparameters, "validationRatio",   0)
        )
    end 

    # Asignamos la matriz de entradas (inputs) y el vector de salidas deseadas (targets)
    inputs, targets = dataset

    # Obtener el número total de clases
    classes = unique(targets)  
    numClasses = length(classes)

    # Calcular el número de folds
    folds = maximum(crossValidationIndices)
    # Inicializar vectores para cada métrica de clasificación
    metricsName = (:acc, :errorRate, :recall, :specificity, :precision, :NPV, :f1)  # Crea una tupla con cada métrica de clasificación de tipo Symbol (más eficiente como clave)
    metricsFold = Dict(metric => zeros(folds) for metric in metricsName)  # Crea un diccionario con pares clave => valor para cada metric en la tupla metrics
    # Inicializar la matriz de confusión con valores reales iguales a 0
    confusionMatrixTotal = zeros(Float64, numClasses, numClasses)

    # BUCLE DE VALIDACIÓN CRUZADA
    for fold in 1:folds
        # Variables de entrada y salida deseada para entrenamiento
        trainInputs = inputs[crossValidationIndices .!= fold, :]
        trainTargets =  targets[crossValidationIndices .!= fold]
        # Variables de entrada y salida deseada para test
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = targets[crossValidationIndices .== fold]


        if modelType == :DoME  # Si utilizamos el algoritmo de DoME, simplemente llamamos a la función desarrollada en el ejercicio 4
            testOutputs = trainClassDoME(
                (trainInputs, trainTargets), 
                testInputs, 
                get(modelHyperparameters, "maximumNodes", 0)
            )
        # Utilizamos un modelo de la librería MLJ y asignamos sus hiperparámetros
        else  
            # Transformar el vector de salidas deseadas en un vector de cadenas de texto para evitar cualquier posible error con las distintas librerías propias de los modelos
            targets = string.(targets)  
            
            if modelType == :SVC
                kernelMap = Dict(
                    "linear"  => LIBSVM.Kernel.Linear,
                    "rbf"     => LIBSVM.Kernel.RadialBasis,
                    "sigmoid" => LIBSVM.Kernel.Sigmoid,
                    "poly"    => LIBSVM.Kernel.Polynomial
                )
                model = SVMClassifier(
                    kernel = kernelMap[get(modelHyperparameters, "kernel", "rbf")],
                    cost = Float64(get(modelHyperparameters, "cost", 1.)),  # Cost se utiliza en todos los casos
                    gamma  = Float64(get(modelHyperparameters, "gamma", 2.)), 
                    degree = Int32(get(modelHyperparameters, "degree", 3)),
                    coef0  = Float64(get(modelHyperparameters, "coef0", 1.))
                )
            
            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(
                    max_depth = get(modelHyperparameters, "max_depth", 4), 
                    rng       = Random.MersenneTwister(1)
                )
            
            elseif modelType == :KNeighborsClassifier
                model = kNNClassifier(
                    K = get(modelHyperparameters, "n_neighbors", 3)
                ) 
            end

            mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))  # Creamos el objeto machine
            MLJ.fit!(mach, verbosity=0)  # Entrenamiento del objeto machine
            
            # La función predict nos permite realizar predicciones sobre los datos 
            testOutputs = MLJ.predict(mach, MLJ.table(testInputs))

            # Para árboles de decisión y kNN 
            # Convertimos los resultados probabilísticos a categóricos para poder compararla con las salidas deseadas
            if modelType in (:DecisionTreeClassifier, :KNeighborsClassifier)
                testOutputs = mode.(testOutputs)
            end
        end

        # Obtenemos los resultados de clasificación y la matriz de confusión
        acc, errorRate, recall, specificity, precision, NPV, f1, confusionMatrixFold = confusionMatrix(testOutputs, testTargets, classes)
        # Asignamos los resultados de las métricas obtenidas al array correspondiente
        metricsFold[:acc][fold]         = acc
        metricsFold[:errorRate][fold]   = errorRate
        metricsFold[:recall][fold]      = recall
        metricsFold[:specificity][fold] = specificity
        metricsFold[:precision][fold]   = precision
        metricsFold[:NPV][fold]         = NPV
        metricsFold[:f1][fold]          = f1
        # Sumamos la matriz de confusión obtenida a la matriz de confusión global en test
        confusionMatrixTotal += confusionMatrixFold
    end

    return (
        (mean(metricsFold[:acc]),         std(metricsFold[:acc])),
        (mean(metricsFold[:errorRate]),   std(metricsFold[:errorRate])),
        (mean(metricsFold[:recall]),      std(metricsFold[:recall])),
        (mean(metricsFold[:specificity]), std(metricsFold[:specificity])),
        (mean(metricsFold[:precision]),   std(metricsFold[:precision])),
        (mean(metricsFold[:NPV]),         std(metricsFold[:NPV])),
        (mean(metricsFold[:f1]),          std(metricsFold[:f1])),
        confusionMatrixTotal
    )
end;
