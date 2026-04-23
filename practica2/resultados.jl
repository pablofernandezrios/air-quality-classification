using DelimitedFiles
using Statistics

# Cargamos las funciones realizadas en la práctica 1
include("firmas.jl")

# Cargamos el dataset
dataset = readdlm("pollution.csv",',')

# Eliminar outliers físicamente imposibles por columna
# Columnas: Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity_to_Industrial_Areas, Population_Density
inputs = Float32.(dataset[2:end, 1:9])  # Omitimos la cabecera con los nombres de las columnas
limits = [
#   (min,    max  )
    (-30.0,  60.0),   # Temperature (°C)
    (0.0,   100.0),   # Humidity (%)
    (0.0,   500.0),   # PM2.5 (µg/m³)
    (0.0,   500.0),   # PM10 (µg/m³)
    (0.0,   200.0),   # NO2 (µg/m³)
    (0.0,   200.0),   # SO2 (µg/m³)
    (0.0,    50.0),   # CO (mg/m³)
    (0.0,   100.0),   # Proximity_to_Industrial_Areas (km)
    (0.0, 50000.0),   # Population_Density (hab/km²)
]

mask = ones(Bool, size(inputs, 1))
for (col, (low, high)) in enumerate(limits)
    mask .&= (inputs[:, col] .>= low) .& (inputs[:, col] .<= high)
end
println("Filas eliminadas por límites físicos: ", sum(.!mask), " de ", size(inputs, 1))

# Preparamos las entradas (aplicando la máscara de outliers)
inputs = inputs[mask, :]
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas: guardamos el vector categórico original (omitiendo cabecera y aplicando máscara)
targetsRaw = dataset[2:end, 10][mask]
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targetsRaw), " de tipo ", typeof(targetsRaw));
classes = unique(targetsRaw)

# Transformamos la variable objetivo (categórica) a numérica para poder calcular índices de validación cruzada
targets = oneHotEncoding(targetsRaw, classes)
println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

# Comprobamos que ambas matrices tienen el mismo número de filas
@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"


# NORMALIZACIÓN DE LOS DATOS
# Usando normalizacion Min-Max
normalizationParameters = calculateMinMaxNormalizationParameters(inputs)
newInputs = normalizeMinMax!(inputs, normalizationParameters)

# # Usando normalizacion Zero mean
# normalizationParameters = calculateZeroMeanNormalizationParameters(inputs)
# newInputs = normalizeZeroMean!(dataset=inputs, normalizationParameters=normalizationParameters)

# Calculamos el número de índices con un valor k=5
crossValidationIndices = crossvalidation(targets, 5)

# El dataset para entrenar usa el vector categórico original (las funciones codifican internamente)
dataset = (newInputs, targetsRaw)

# Función para imprimir los resultados de cada modelo
function printModelResults(name::String, metrics, classes)
    (_, _, _, _, _, _, f1Score, confMatrix) = metrics
    println("\n", "="^50)
    println(" Modelo: $name")
    println("="^50)
    println("\nMétricas por clase:")
    for (i, class) in enumerate(classes)
        TP = confMatrix[i, i]
        FP = sum(confMatrix[:, i]) - TP
        FN = sum(confMatrix[i, :]) - TP
        prec = TP / (TP + FP)
        rec  = TP / (TP + FN)
        f1   = 2 * prec * rec / (prec + rec)
        println("  [$class]  Precision: $(round(prec, digits=3))  Recall: $(round(rec, digits=3))  F1: $(round(f1, digits=3))")
    end
    println("\nF1-score global: $(round(f1Score[1], digits=3)) ± $(round(f1Score[2], digits=3))")
    println("\nMatriz de confusión:")
    for row in eachrow(confMatrix)
        println("  ", join(lpad.(row, 6), "  "))
    end
end

# Red de Neuronas Artificial
topology = [10, 5]
metricsANN = ANNCrossValidation(topology, dataset, crossValidationIndices)
printModelResults("ANN", metricsANN, classes)

# SVM
modelHyperparameters = Dict("kernel" => "linear",
                            "C"      => 1.0,
                            "gamma"  => -1.0,
                            "degree" => -1,
                            "coef0"  => -1.0)
metricsSVM = modelCrossValidation(:SVC, modelHyperparameters, dataset, crossValidationIndices)
printModelResults("SVM", metricsSVM, classes)

# KNN
modelHyperparameters = Dict("n_neighbors" => 5)
metricsKNN = modelCrossValidation(:KNeighborsClassifier, modelHyperparameters, dataset, crossValidationIndices)
printModelResults("KNN", metricsKNN, classes)

# Árbol de decisión
modelHyperparameters = Dict("max_depth" => 10)
metricsDT = modelCrossValidation(:DecisionTreeClassifier, modelHyperparameters, dataset, crossValidationIndices)
printModelResults("Decision Tree", metricsDT, classes)
