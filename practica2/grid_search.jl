# ==============================================================================
# grid_search.jl
# ------------------------------------------------------------------------------
# Búsqueda paralela de hiperparámetros sobre los cuatro modelos del proyecto
# (ANN, SVM, KNN, Decision Tree). Coexiste con el resultados.jl original sin
# modificarlo: este script es totalmente independiente y reutiliza únicamente
# las funciones definidas en firmas.jl.
#
# Uso:
#   julia --threads 4 grid_search.jl
#   JULIA_NUM_WORKERS=8 julia --threads 4 grid_search.jl   # forzar nº de workers
#
# El flag --threads controla cuántas configs de ANN se entrenan en paralelo sobre
# la GPU. Con --threads 1 (defecto) la ANN corre secuencialmente.
# ==============================================================================

using Distributed
using CUDA

# Silenciamos el warning de MLDataDevices "No functional GPU backend found"
# que aparece porque no cargamos cuDNN. No lo necesitamos: usamos cu() directo.
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# ----------------------------------------------------------------------------------------------
# Configuración del paralelismo
# ----------------------------------------------------------------------------------------------
# Por defecto usamos todos los cores físicos menos uno (para dejar el sistema responsivo).
# Se puede ajustar con la variable de entorno JULIA_NUM_WORKERS.
const N_WORKERS = parse(Int, get(ENV, "JULIA_NUM_WORKERS", string(max(1, Sys.CPU_THREADS - 1))))

let extra = nprocs() - 1
    if extra < N_WORKERS
        addprocs(N_WORKERS - extra; exeflags="--project")
    end
end

println("Workers activos: ", nworkers(), " (cores físicos: ", Sys.CPU_THREADS, ")")
println("Threads Julia:   ", Threads.nthreads(), " (para ANN en GPU; lanza con --threads N para aumentar)")

# ----------------------------------------------------------------------------------------------
# Carga de paquetes y funciones en TODOS los workers
# ----------------------------------------------------------------------------------------------
@everywhere using DelimitedFiles
@everywhere using Statistics
@everywhere include("firmas2.jl")

# ----------------------------------------------------------------------------------------------
# Carga del dataset y preprocesado (solo en el driver)
# ----------------------------------------------------------------------------------------------
dataset = readdlm("pollution.csv", ',')

# Eliminar outliers físicamente imposibles por columna
inputs = Float32.(dataset[2:end, 1:9])
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
println("Filas eliminadas por límites físicos: ", sum(.!mask), " de ", size(inputs, 1))

inputs = inputs[mask, :]
println("Tamaño de la matriz de entradas: ", size(inputs, 1), "x", size(inputs, 2))

targetsRaw = dataset[2:end, 10][mask]
classes = unique(targetsRaw)

targets = oneHotEncoding(targetsRaw, classes)
@assert (size(inputs, 1) == size(targets, 1))

# Normalización Min-Max
normalizationParameters = calculateMinMaxNormalizationParameters(inputs)
newInputs = normalizeMinMax!(inputs, normalizationParameters)

const NUM_FOLDS = 5
crossValidationIndices = crossvalidation(targets, NUM_FOLDS)

# Las funciones *CrossValidation aceptan el target en formato categórico
dataset_tuple = (newInputs, targetsRaw)

# ----------------------------------------------------------------------------------------------
# Difusión de datos a los workers
# ----------------------------------------------------------------------------------------------
# Hacemos disponibles en cada worker las variables que las tareas paralelas necesitan leer.
# La interpolación $ con @everywhere envía el valor del driver a cada worker y lo asigna
# como variable global en su Main. Evita reenviar el dataset (~MB) en cada llamada de pmap.
@everywhere dataset_tuple = $dataset_tuple
@everywhere crossValidationIndices = $crossValidationIndices
@everywhere classes = $classes

# ----------------------------------------------------------------------------------------------
# Definición de los grids de hiperparámetros
# ----------------------------------------------------------------------------------------------
# ANN: 8 arquitecturas (4 con 1 capa oculta, 4 con 2 capas ocultas), lr y epochs fijos
const ANN_TOPOLOGIES     = [[10], [20], [30], [50],
                             [10, 5], [20, 10], [30, 15], [50, 25]]
const ANN_LEARNING_RATES = [0.01]
const ANN_MAX_EPOCHS     = [1000]
const ANN_NUM_EXECUTIONS = 5

const ANN_GRID = [
    Dict("topology" => topo, "learningRate" => lr, "maxEpochs" => me, "numExecutions" => ANN_NUM_EXECUTIONS)
    for topo in ANN_TOPOLOGIES, lr in ANN_LEARNING_RATES, me in ANN_MAX_EPOCHS
] |> vec

# SVM: 8 configuraciones (4 lineales + 4 RBF) variando C y gamma
const SVM_GRID = let grid = Dict[]
    for C in [0.1, 1.0, 10.0, 100.0]
        push!(grid, Dict("kernel" => "linear", "C" => C, "gamma" => -1.0, "degree" => -1, "coef0" => -1.0))
    end
    for (C, γ) in [(1.0, 0.1), (1.0, 1.0), (10.0, 0.1), (10.0, 1.0)]
        push!(grid, Dict("kernel" => "rbf", "C" => C, "gamma" => γ, "degree" => -1, "coef0" => -1.0))
    end
    grid
end

# DoME: 8 valores de maximumNodes en el rango típicamente útil
const DOME_GRID = [Dict("maximumNodes" => n) for n in [5, 10, 15, 20, 30, 50, 75, 100]]

# KNN: 6 valores de k
const KNN_GRID = [Dict("n_neighbors" => k) for k in [3, 5, 7, 11, 15, 21]]

# DT: 6 profundidades
const DT_GRID = [Dict("max_depth" => d) for d in [3, 5, 7, 10, 15, 20]]

println("\nTamaño de los grids:")
println("  ANN:  ", length(ANN_GRID),  " configuraciones")
println("  SVM:  ", length(SVM_GRID),  " configuraciones")
println("  DoME: ", length(DOME_GRID), " configuraciones")
println("  KNN:  ", length(KNN_GRID),  " configuraciones")
println("  DT:   ", length(DT_GRID),   " configuraciones")

# ----------------------------------------------------------------------------------------------
# Funciones de evaluación remotas
# ----------------------------------------------------------------------------------------------
@everywhere function evaluateANNConfig(hp::Dict)
    topology = hp["topology"]
    metrics = ANNCrossValidation(topology, dataset_tuple, crossValidationIndices;
        numExecutions = hp["numExecutions"],
        learningRate  = hp["learningRate"],
        maxEpochs     = hp["maxEpochs"])
    return (hp, metrics)
end

@everywhere function evaluateClassicConfig(modelType::Symbol, hp::Dict)
    metrics = modelCrossValidation(modelType, hp, dataset_tuple, crossValidationIndices)
    return (hp, metrics)
end

# ----------------------------------------------------------------------------------------------
# Selección por F1-score
# ----------------------------------------------------------------------------------------------
function bestByF1(results)
    bestIdx = argmax([r[2][7][1] for r in results])
    return bestIdx, results[bestIdx]
end

# Reutiliza confusionMatrix(...; weighted=false) de firmas2.jl para obtener métricas macro
# desde la matriz de confusión agregada que devuelven las funciones *CrossValidation.
function macroMetrics(aggregatedConfMatrix::AbstractMatrix)
    counts = round.(Int, aggregatedConfMatrix)
    total = sum(counts)
    n = size(counts, 1)
    targets = falses(total, n)
    outputs = falses(total, n)
    idx = 1
    for i in 1:n, j in 1:n
        c = counts[i, j]
        if c > 0
            targets[idx:idx+c-1, i] .= true
            outputs[idx:idx+c-1, j] .= true
            idx += c
        end
    end
    return confusionMatrix(outputs, targets; weighted=false)
end

# ----------------------------------------------------------------------------------------------
# Impresión de resultados (con todas las métricas)
# ----------------------------------------------------------------------------------------------
function printModelResults(name::String, hp::Dict, metrics, classes)
    (acc, errRate, recall, spec, prec, npv, f1, confMatrix) = metrics
    println("\n", "="^60)
    println(" Mejor configuración de $name (por F1 promedio)")
    println("="^60)
    println("\nHiperparámetros:")
    for (k, v) in hp
        println("  $k = $v")
    end
    println("\nMétricas globales (media ± desviación sobre folds):")
    println("  Accuracy   : $(round(acc[1],     digits=4)) ± $(round(acc[2],     digits=4))")
    println("  Error rate : $(round(errRate[1], digits=4)) ± $(round(errRate[2], digits=4))")
    println("  Recall     : $(round(recall[1],  digits=4)) ± $(round(recall[2],  digits=4))")
    println("  Specificity: $(round(spec[1],    digits=4)) ± $(round(spec[2],    digits=4))")
    println("  Precision  : $(round(prec[1],    digits=4)) ± $(round(prec[2],    digits=4))")
    println("  NPV          : $(round(npv[1],     digits=4)) ± $(round(npv[2],     digits=4))")
    println("  F1 ponderado : $(round(f1[1],      digits=4)) ± $(round(f1[2],      digits=4))")
    println("  F1 macro     : $(round(macroMetrics(confMatrix)[7], digits=4))   (sin std; sobre matriz agregada)")
    println("\nMétricas por clase:")
    for (i, class) in enumerate(classes)
        TP = confMatrix[i, i]
        FP = sum(confMatrix[:, i]) - TP
        FN = sum(confMatrix[i, :]) - TP
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f = 2 * p * r / (p + r)
        println("  [$class]  Precision: $(round(p, digits=3))  Recall: $(round(r, digits=3))  F1: $(round(f, digits=3))")
    end
    println("\nMatriz de confusión:")
    for row in eachrow(confMatrix)
        println("  ", join(lpad.(row, 6), "  "))
    end
end

function hpSummary(hp::Dict)::String
    if haskey(hp, "topology")
        return "topo=$(hp["topology"]) lr=$(hp["learningRate"]) epochs=$(hp["maxEpochs"])"
    elseif haskey(hp, "kernel")
        s = "kernel=$(hp["kernel"]) C=$(hp["C"])"
        hp["kernel"] == "rbf"  && (s *= " γ=$(hp["gamma"])")
        hp["kernel"] == "poly" && (s *= " d=$(hp["degree"]) c0=$(hp["coef0"])")
        return s
    elseif haskey(hp, "maximumNodes")
        return "maximumNodes=$(hp["maximumNodes"])"
    elseif haskey(hp, "n_neighbors")
        return "k=$(hp["n_neighbors"])"
    elseif haskey(hp, "max_depth")
        return "max_depth=$(hp["max_depth"])"
    else
        return join(["$k=$v" for (k,v) in hp], " ")
    end
end

function printAllResults(name::String, results)
    println("\n", "="^100)
    println(" Todas las configuraciones de $name (ordenadas por F1 ponderado desc)")
    println("="^100)
    col1 = 52
    header = rpad("Hiperparámetros", col1) * " │ " *
             lpad("Acc",   6) * "  " * lpad("F1pond", 7) * "  " * lpad("F1mac", 6) * "  " *
             lpad("Prec",  6) * "  " * lpad("Recall", 6) * "  " * lpad("std F1", 6)
    println(header)
    println("-"^100)
    sorted = sort(results, by = r -> -r[2][7][1])
    for (hp, metrics) in sorted
        (acc, _, recall, _, prec, _, f1, confMatrix) = metrics
        f1m = macroMetrics(confMatrix)[7]
        row = rpad(hpSummary(hp), col1) * " │ " *
              lpad(round(acc[1],    digits=4), 6) * "  " *
              lpad(round(f1[1],     digits=4), 7) * "  " *
              lpad(round(f1m,       digits=4), 6) * "  " *
              lpad(round(prec[1],   digits=4), 6) * "  " *
              lpad(round(recall[1], digits=4), 6) * "  " *
              lpad(round(f1[2],     digits=4), 6)
        println(row)
    end
    println("-"^100)
end

# Tras cada modelo, imprimimos sus resultados al instante para que estén
# visibles aunque un modelo posterior crashee.
function reportModel(name::String, results, classes)
    flush(stdout)
    _, best = bestByF1(results)
    printModelResults(name, best[1], best[2], classes)
    printAllResults(name, results)
    flush(stdout)
    return best
end

# ----------------------------------------------------------------------------------------------
# EJECUCIÓN DEL GRID SEARCH PARALELO
# ----------------------------------------------------------------------------------------------
println("\n", "#"^60)
println("# Iniciando grid search paralelo")
println("#"^60)

# Orden de ejecución: del más rápido al más lento. Si algo falla a mitad,
# al menos quedan reportados los modelos baratos.

println("\n[1/5] Evaluando Decision Tree ($(length(DT_GRID)) configuraciones)...")
t_dt = @elapsed dt_results = pmap(hp -> evaluateClassicConfig(:DecisionTreeClassifier, hp), DT_GRID)
println("  Tiempo total DT: $(round(t_dt, digits=1)) s")
bestDT = reportModel("Decision Tree", dt_results, classes)

println("\n[2/5] Evaluando KNN ($(length(KNN_GRID)) configuraciones)...")
t_knn = @elapsed knn_results = pmap(hp -> evaluateClassicConfig(:KNeighborsClassifier, hp), KNN_GRID)
println("  Tiempo total KNN: $(round(t_knn, digits=1)) s")
bestKNN = reportModel("KNN", knn_results, classes)

println("\n[3/5] Evaluando SVM ($(length(SVM_GRID)) configuraciones)...")
t_svm = @elapsed svm_results = pmap(hp -> evaluateClassicConfig(:SVC, hp), SVM_GRID)
println("  Tiempo total SVM: $(round(t_svm, digits=1)) s")
bestSVM = reportModel("SVM", svm_results, classes)

println("\n[4/5] Evaluando DoME ($(length(DOME_GRID)) configuraciones)...")
t_dome = @elapsed dome_results = pmap(hp -> evaluateClassicConfig(:DoME, hp), DOME_GRID)
println("  Tiempo total DoME: $(round(t_dome, digits=1)) s")
bestDoME = reportModel("DoME", dome_results, classes)

println("\n[5/5] Evaluando ANN ($(length(ANN_GRID)) configuraciones) en $(USE_GPU ? "GPU" : "CPU") × $(Threads.nthreads()) thread(s)...")
t_ann = @elapsed begin
    ann_results = Vector{Any}(undef, length(ANN_GRID))
    Threads.@threads for i in eachindex(ANN_GRID)
        ann_results[i] = evaluateANNConfig(ANN_GRID[i])
    end
end
println("  Tiempo total ANN: $(round(t_ann, digits=1)) s")
bestANN = reportModel("ANN", ann_results, classes)

println("\nTiempo total del grid search: $(round(t_dt + t_knn + t_svm + t_dome + t_ann, digits=1)) s")

# ----------------------------------------------------------------------------------------------
# (Los reportes individuales de cada modelo ya se imprimieron tras su evaluación
# vía reportModel, para que estén disponibles aunque algún modelo posterior falle.)

# ----------------------------------------------------------------------------------------------
# Comparativa final
# ----------------------------------------------------------------------------------------------
println("\n", "#"^60)
println("# RESUMEN COMPARATIVO (mejor configuración de cada modelo)")
println("#"^60)
println("\nModelo          | F1 ponderado (media ± std)  | F1 macro")
println("-"^65)
for (name, best) in [("ANN", bestANN), ("SVM", bestSVM), ("DoME", bestDoME), ("KNN", bestKNN), ("Decision Tree", bestDT)]
    f1  = best[2][7]
    f1m = macroMetrics(best[2][8])[7]
    println(rpad(name, 16), "| $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4))         | $(round(f1m, digits=4))")
end

ranking = [("ANN", bestANN), ("SVM", bestSVM), ("DoME", bestDoME), ("KNN", bestKNN), ("Decision Tree", bestDT)]
sort!(ranking, by = x -> -x[2][2][7][1])
println("\nGanador global: ", ranking[1][1], " con F1 = ",
        round(ranking[1][2][2][7][1], digits=4))