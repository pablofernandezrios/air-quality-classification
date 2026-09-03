# ==============================================================================
# resultados.jl  —  Archivo Ejecutable Final (Resumen de Mejores Modelos)
# ==============================================================================
# Este script está diseñado para ser ejecutado por el evaluador.
# Contiene el preprocesado de datos (incluyendo análisis de multicolinealidad),
# y entrena las mejores configuraciones de hiperparámetros encontradas en el 
# grid search para los 5 algoritmos estudiados, mostrando métricas y matrices.
# ==============================================================================

using DelimitedFiles
using Statistics
using Printf
using Random

# Rutas del proyecto (independientes del directorio desde el que se lance Julia)
const SRC_DIR  = @__DIR__
const ROOT_DIR = normpath(joinpath(SRC_DIR, ".."))
const DATA_DIR = joinpath(ROOT_DIR, "data")

# Cargamos las funciones realizadas en la práctica 1 y soporte MLJ
include(joinpath(SRC_DIR, "firmas2.jl"))

println("\n" * "*"^60)
println("* EVALUACIÓN FINAL DE MODELOS DE CALIDAD DEL AIRE")
println("*"^60)

# Cargamos el dataset
dataset = readdlm(joinpath(DATA_DIR, "pollution.csv"), ',')

# Eliminar outliers físicamente imposibles por columna
# Columnas: Temp, Hum, PM2.5, PM10, NO2, SO2, CO, Prox_Ind, Pop_Den
inputs = Float32.(dataset[2:end, 1:9])  # Omitimos la cabecera
limits = [
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
println("\nFilas eliminadas por violar límites físicos: ", sum(.!mask), " de ", size(inputs, 1))

# Preparamos las entradas filtradas
inputs = inputs[mask, :]
println("Tamaño final de la matriz de características: ", size(inputs,1), "x", size(inputs,2))

# ==============================================================================
# NUEVO: Análisis de Correlación de Pearson (Reducción de Dimensionalidad)
# ==============================================================================
println("\n" * "="^60)
println(" ANÁLISIS DE CORRELACIÓN DE PEARSON (Multicolinealidad)")
println("="^60)

colNames = ["Temp", "Hum", "PM2.5", "PM10", "NO2", "SO2", "CO", "ProxInd", "PopDen"]
corMatrix = cor(inputs)

# Imprimir cabecera
print("        ")
for name in colNames
    @printf("%8s", name)
end
println()

# Imprimir filas de la matriz
for i in 1:size(corMatrix, 1)
    @printf("%8s", colNames[i])
    for j in 1:size(corMatrix, 2)
        @printf("%8.2f", corMatrix[i, j])
    end
    println()
end
println("------------------------------------------------------------")
println("Nota: Observar las correlaciones entre contaminantes (ej. PM2.5 y PM10)")
println("para justificar posibles reducciones de dimensionalidad en la memoria.")

# ==============================================================================
# Preprocesado de la Variable Objetivo y Normalización
# ==============================================================================
targetsRaw = dataset[2:end, 10][mask]
classes = unique(targetsRaw)

# Transformamos la variable categórica a numérica (One-Hot) para crossvalidation
targets = oneHotEncoding(targetsRaw, classes)

# Normalización Min-Max (Decisión de diseño justificada en la memoria)
normalizationParameters = calculateMinMaxNormalizationParameters(inputs)
newInputs = normalizeMinMax!(inputs, normalizationParameters)

# Reproducibilidad
Random.seed!(67) # Fijamos la semilla para la inicialización estocástica (ANN, etc.)

const NUM_FOLDS   = 5
const INDICES_CSV = joinpath(DATA_DIR, "indices.csv")
if isfile(INDICES_CSV)
    println("\n[Reproducibilidad] Cargando índices de validación cruzada desde 'data/indices.csv'...")
    crossValidationIndices = Int.(vec(readdlm(INDICES_CSV, ',')))
else
    println("\n[Reproducibilidad] Generando índices estratificados y guardándolos en 'data/indices.csv'...")
    crossValidationIndices = crossvalidation(targets, NUM_FOLDS)
    writedlm(INDICES_CSV, crossValidationIndices, ',')
end

# Dataset preparado para MLJ y funciones personalizadas
dataset_tup = (newInputs, targetsRaw)

# Función estandarizada para imprimir los resultados
function printModelResults(name::String, metrics, classes)
    (_, _, _, _, _, _, f1Score, confMatrix) = metrics
    println("\n", "="^50)
    println(" Modelo: $name (Mejores Hiperparámetros)")
    println("="^50)
    println("\nMétricas por clase:")
    for (i, class) in enumerate(classes)
        TP = confMatrix[i, i]
        FP = sum(confMatrix[:, i]) - TP
        FN = sum(confMatrix[i, :]) - TP
        prec = TP + FP > 0 ? TP / (TP + FP) : 0.0
        rec  = TP + FN > 0 ? TP / (TP + FN) : 0.0
        f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0.0
        println("  [$(rpad(string(class), 10))]  Precision: $(round(prec, digits=3))  Recall: $(round(rec, digits=3))  F1: $(round(f1, digits=3))")
    end
    println("\nF1-score global (media ± std): $(round(f1Score[1], digits=4)) ± $(round(f1Score[2], digits=4))")
    println("\nMatriz de confusión:")
    for row in eachrow(confMatrix)
        println("  ", join(lpad.(row, 6), "  "))
    end
end

# ==============================================================================
# ENTRENAMIENTO Y EVALUACIÓN (Mejores configuraciones)
# ==============================================================================
# Nota: Si el Grid Search encuentra otros parámetros mejores, cámbialos aquí.

# 1. Árbol de decisión
modelHyperparameters_DT = Dict("max_depth" => 10)
println("\n[1/5] Entrenando Decision Tree...")
metricsDT = @time modelCrossValidation(:DecisionTreeClassifier, modelHyperparameters_DT, dataset_tup, crossValidationIndices)
printModelResults("Decision Tree", metricsDT, classes)

# 2. KNN
modelHyperparameters_KNN = Dict("n_neighbors" => 5)
println("\n[2/5] Entrenando KNN...")
metricsKNN = @time modelCrossValidation(:KNeighborsClassifier, modelHyperparameters_KNN, dataset_tup, crossValidationIndices)
printModelResults("K-Nearest Neighbors", metricsKNN, classes)

# 3. SVM 
modelHyperparameters_SVM = Dict(
    "kernel" => "rbf", 
    "C"      => 10.0,
    "gamma"  => 0.1, 
    "degree" => -1, 
    "coef0"  => -1.0
)
println("\n[3/5] Entrenando SVM...")
metricsSVM = @time modelCrossValidation(:SVC, modelHyperparameters_SVM, dataset_tup, crossValidationIndices)
printModelResults("Support Vector Machine (SVM)", metricsSVM, classes)

# 4. DoME (¡Arreglado el error de sintaxis y de impresión!)
modelHyperparameters_DoME = Dict("maximumNodes" => 100)
println("\n[4/5] Entrenando DoME (Regresión Simbólica)...")
metricsDoME = @time modelCrossValidation(:DoME, modelHyperparameters_DoME, dataset_tup, crossValidationIndices)
printModelResults("DoME", metricsDoME, classes)

# 5. Red de Neuronas Artificial (ANN)
topology = [20, 10]
println("\n[5/5] Entrenando ANN (GPU activada si disponible)...")
metricsANN = @time ANNCrossValidation(topology, dataset_tup, crossValidationIndices; numExecutions=5)
printModelResults("Artificial Neural Network (ANN)", metricsANN, classes)

println("\n", "*"^60)
println("* EJECUCIÓN FINALIZADA CON ÉXITO")
println("*"^60)