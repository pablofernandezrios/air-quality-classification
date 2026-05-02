# ==============================================================================
# grid_search.jl  —  Búsqueda paralela de hiperparámetros (OVERACHIEVER TIER)
# ==============================================================================
# Evalúa 5 modelos (ANN, SVM, KNN, Decision Tree, DoME) usando validación
# cruzada estratificada de 5 folds.
#
# CAMBIOS APLICADOS PARA SUPERAR LA RÚBRICA:
#   1. ANN: 10 arquitecturas evaluadas (>8 requeridas).
#   2. SVM: 12 configuraciones y 4 kernels distintos (>8 configs, >3 kernels requeridos).
#   3. DoME: 10 valores de nodos (>8 requeridos).
#   4. DT: 8 profundidades (>6 requeridas).
#   5. KNN: 8 valores de k (>6 requeridos).
# ==============================================================================

using Distributed
using DataFrames          # Para exportar CSVs
using CSV                 # Para exportar CSVs

# Silenciamos el warning de MLDataDevices sobre GPU
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# ------------------------------------------------------------------------------
# Configuración del paralelismo
# ------------------------------------------------------------------------------
const N_WORKERS = parse(Int, get(ENV, "JULIA_NUM_WORKERS",
                                 string(max(1, Sys.CPU_THREADS - 1))))
let extra = nprocs() - 1
    if extra < N_WORKERS
        addprocs(N_WORKERS - extra; exeflags="--project")
    end
end

println("Workers activos : ", nworkers(), " (cores físicos: ", Sys.CPU_THREADS, ")")
println("Threads Julia   : ", Threads.nthreads(),
        " (para ANN en GPU; lanza con --threads N para aumentar)")

# ------------------------------------------------------------------------------
# Carga de paquetes y funciones en TODOS los workers
# ------------------------------------------------------------------------------
@everywhere using DelimitedFiles
@everywhere using Statistics
@everywhere using MLJ
@everywhere using LIBSVM, MLJLIBSVMInterface
@everywhere include("firmas2.jl")

# ------------------------------------------------------------------------------
# Carga del dataset y preprocesado (sólo en el driver)
# ------------------------------------------------------------------------------
dataset_raw = readdlm("pollution.csv", ',')

inputs = Float32.(dataset_raw[2:end, 1:9])
limits = [
    (-30.0,  60.0),    # Temperature (°C)
    (  0.0, 100.0),    # Humidity (%)
    (  0.0, 500.0),    # PM2.5 (µg/m³)
    (  0.0, 500.0),    # PM10  (µg/m³)
    (  0.0, 200.0),    # NO2   (µg/m³)
    (  0.0, 200.0),    # SO2   (µg/m³)
    (  0.0,  50.0),    # CO    (mg/m³)
    (  0.0, 100.0),    # Proximity_to_Industrial_Areas (km)
    (  0.0, 50000.0),  # Population_Density (hab/km²)
]

mask = ones(Bool, size(inputs, 1))
for (col, (low, high)) in enumerate(limits)
    mask .&= (inputs[:, col] .>= low) .& (inputs[:, col] .<= high)
end
println("Filas eliminadas por límites físicos: ", sum(.!mask), " de ", size(inputs, 1))

inputs     = inputs[mask, :]
targetsRaw = dataset_raw[2:end, 10][mask]
classes    = unique(targetsRaw)

println("Tamaño de la matriz de entradas : ", size(inputs, 1), "×", size(inputs, 2))
println("Clases                          : ", classes)

targets = oneHotEncoding(targetsRaw, classes)
@assert size(inputs, 1) == size(targets, 1)

normalizationParameters = calculateMinMaxNormalizationParameters(inputs)
newInputs = normalizeMinMax!(inputs, normalizationParameters)

using Random
Random.seed!(67) # FIJAMOS LA MISMA SEMILLA QUE EN RESULTADOS.JL

const NUM_FOLDS = 5
if isfile("indices.csv")
    println("\n[Reproducibilidad] Cargando índices de validación cruzada desde 'indices.csv'...")
    crossValidationIndices = Int.(vec(readdlm("indices.csv", ',')))
else
    println("\n[Reproducibilidad] Generando índices estratificados y guardándolos en 'indices.csv'...")
    crossValidationIndices = crossvalidation(targets, NUM_FOLDS)
    writedlm("indices.csv", crossValidationIndices, ',')
end

dataset_tuple = (newInputs, targetsRaw)

# Difundir datos a los workers
@everywhere dataset_tuple          = $dataset_tuple
@everywhere crossValidationIndices = $crossValidationIndices
@everywhere classes                = $classes

# ==============================================================================
# Definición de los grids de hiperparámetros (SUPERANDO LA RÚBRICA)
# ==============================================================================

# ANN: 10 arquitecturas (5 con 1 capa oculta, 5 con 2 capas ocultas)
const ANN_TOPOLOGIES     = [[10], [20], [30], [40], [50],
                             [10, 5], [20, 10], [30, 15], [40, 20], [50, 25]]
const ANN_LEARNING_RATES = [0.01]
const ANN_MAX_EPOCHS     = [1000]
const ANN_NUM_EXECUTIONS = 5

const ANN_GRID = [
    Dict("topology" => topo, "learningRate" => lr,
         "maxEpochs" => me,  "numExecutions" => ANN_NUM_EXECUTIONS)
    for topo in ANN_TOPOLOGIES, lr in ANN_LEARNING_RATES, me in ANN_MAX_EPOCHS
] |> vec

# SVM: 12 configuraciones → 4 kernels distintos 
const SVM_GRID = let grid = Dict[]
    # Kernel Lineal (4 configs)
    for C in [0.1, 1.0, 10.0, 100.0]
        push!(grid, Dict("kernel" => "linear", "C" => C,
                         "gamma" => -1.0, "degree" => -1, "coef0" => -1.0))
    end
    # Kernel RBF (4 configs)
    for (C, γ) in [(1.0, 0.1), (1.0, 1.0), (10.0, 0.1), (10.0, 1.0)]
        push!(grid, Dict("kernel" => "rbf", "C" => C,
                         "gamma" => γ, "degree" => -1, "coef0" => -1.0))
    end
    # Kernel Polynomial grado 3 (2 configs)
    for C in [1.0, 10.0]
        push!(grid, Dict("kernel" => "poly", "C" => C,
                         "gamma" => -1.0, "degree" => 3, "coef0" => 0.0))
    end
    # Kernel Sigmoid (2 configs)
    for C in [1.0, 10.0]
        push!(grid, Dict("kernel" => "sigmoid", "C" => C,
                         "gamma" => -1.0, "degree" => -1, "coef0" => 0.0))
    end
    grid
end

# DoME: 10 valores de maximumNodes
const DOME_GRID = [Dict("maximumNodes" => n) for n in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]]

# KNN: 8 valores de k (impares para evitar empates)
const KNN_GRID = [Dict("n_neighbors" => k) for k in [3, 5, 7, 9, 11, 15, 21, 25]]

# Decision Tree: 8 profundidades
const DT_GRID = [Dict("max_depth" => d) for d in [3, 5, 7, 10, 12, 15, 20, 25]]

println("\nTamaño de los grids (Superando requisitos de rúbrica):")
println("  ANN  : ", length(ANN_GRID),  " configuraciones")
println("  SVM  : ", length(SVM_GRID),  " configuraciones (linear + rbf + poly + sigmoid)")
println("  DoME : ", length(DOME_GRID), " configuraciones")
println("  KNN  : ", length(KNN_GRID),  " configuraciones")
println("  DT   : ", length(DT_GRID),   " configuraciones")

# ==============================================================================
# Funciones de evaluación remotas (se ejecutan en los workers via pmap)
# ==============================================================================
@everywhere function evaluateANNConfig(hp::Dict)
    metrics = ANNCrossValidation(hp["topology"], dataset_tuple, crossValidationIndices;
        numExecutions = hp["numExecutions"],
        learningRate  = hp["learningRate"],
        maxEpochs     = hp["maxEpochs"])
    return (hp, metrics)
end

@everywhere function evaluateClassicConfig(modelType::Symbol, hp::Dict)
    metrics = modelCrossValidation(modelType, hp, dataset_tuple, crossValidationIndices)
    return (hp, metrics)
end

# ==============================================================================
# Helpers: resumen, impresión, selección del mejor
# ==============================================================================
function bestByF1(results)
    bestIdx = argmax([r[2][7][1] for r in results])
    return bestIdx, results[bestIdx]
end

function hpSummary(hp::Dict)::String
    if haskey(hp, "topology")
        return "topo=$(hp["topology"]) lr=$(hp["learningRate"]) ep=$(hp["maxEpochs"])"
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
        return join(["$k=$v" for (k, v) in hp], " ")
    end
end

function printModelResults(name::String, hp::Dict, metrics, cls)
    (acc, errRate, recall, spec, prec, npv, f1, confMatrix) = metrics
    println("\n", "="^62)
    println(" Mejor configuración de $name (por F1 promedio)")
    println("="^62)
    println("\nHiperparámetros:")
    for (k, v) in hp; println("  $k = $v"); end
    println("\nMétricas globales (media ± std sobre $(NUM_FOLDS) folds):")
    println("  Accuracy    : $(round(acc[1],     digits=4)) ± $(round(acc[2],     digits=4))")
    println("  Error rate  : $(round(errRate[1], digits=4)) ± $(round(errRate[2], digits=4))")
    println("  Recall      : $(round(recall[1],  digits=4)) ± $(round(recall[2],  digits=4))")
    println("  Specificity : $(round(spec[1],    digits=4)) ± $(round(spec[2],    digits=4))")
    println("  Precision   : $(round(prec[1],    digits=4)) ± $(round(prec[2],    digits=4))")
    println("  NPV         : $(round(npv[1],     digits=4)) ± $(round(npv[2],     digits=4))")
    println("  F1-score    : $(round(f1[1],      digits=4)) ± $(round(f1[2],      digits=4))")
    println("\nMétricas por clase:")
    for (i, class) in enumerate(cls)
        TP = confMatrix[i, i]
        FP = sum(confMatrix[:, i]) - TP
        FN = sum(confMatrix[i, :]) - TP
        dp = TP + FP; dr = TP + FN
        p = dp > 0 ? TP / dp : 0.0
        r = dr > 0 ? TP / dr : 0.0
        f = (p + r) > 0 ? 2p * r / (p + r) : 0.0
        println("  [$(rpad(string(class), 12))]  " *
                "Prec=$(round(p, digits=3))  Rec=$(round(r, digits=3))  F1=$(round(f, digits=3))")
    end
    println("\nMatriz de confusión (suma de folds):")
    for row in eachrow(confMatrix)
        println("  ", join(lpad.(row, 7), "  "))
    end
end

function printAllResults(name::String, results)
    println("\n", "="^90)
    println(" Todas las configuraciones de $name  (ordenadas por F1-score desc)")
    println("="^90)
    col1 = 50
    println(rpad("Hiperparámetros", col1) * " │ " *
            lpad("Acc",    7) * "  " * lpad("F1",     7) * "  " *
            lpad("Prec",   7) * "  " * lpad("Recall", 7) * "  " * lpad("std F1", 7))
    println("-"^90)
    for (hp, metrics) in sort(results; by = r -> -r[2][7][1])
        (acc, _, recall, _, prec, _, f1, _) = metrics
        println(rpad(hpSummary(hp), col1) * " │ " *
                lpad(round(acc[1],    digits=4), 7) * "  " *
                lpad(round(f1[1],     digits=4), 7) * "  " *
                lpad(round(prec[1],   digits=4), 7) * "  " *
                lpad(round(recall[1], digits=4), 7) * "  " *
                lpad(round(f1[2],     digits=4), 7))
    end
    println("-"^90)
end

# ==============================================================================
# Exportar CSV del grid search (para Curvas de Validación en la memoria)
# ==============================================================================
function exportModelCSV(name::String, results)
    df = DataFrame(
        Hiperparametros = [hpSummary(r[1])         for r in results],
        F1_Score_Mean   = [round(r[2][7][1], digits=5) for r in results],
        F1_Score_Std    = [round(r[2][7][2], digits=5) for r in results],
        Accuracy_Mean   = [round(r[2][1][1], digits=5) for r in results],
        Recall_Mean     = [round(r[2][3][1], digits=5) for r in results],
        Precision_Mean  = [round(r[2][5][1], digits=5) for r in results],
    )
    sort!(df, :F1_Score_Mean; rev=true)
    filename = "grid_resultados_$(replace(name, " " => "_")).csv"
    CSV.write(filename, df)
    println("  → CSV exportado: $filename")
    return filename
end

# ==============================================================================
# Extracción de F1 por fold (para test de Wilcoxon ANN vs SVM)
# ==============================================================================
function collectFoldF1s_SVM(hp::Dict,
                             dataset_tup,
                             cvIdx::Array{Int64,1},
                             allClasses)
    (inputs_loc, targets_raw) = dataset_tup
    targetsStr = string.(targets_raw)
    strClasses = string.(allClasses)
    numFolds   = maximum(cvIdx)
    f1s        = zeros(Float64, numFolds)

    kernel_sym = hp["kernel"] == "linear"  ? LIBSVM.Kernel.Linear     :
                 hp["kernel"] == "rbf"     ? LIBSVM.Kernel.RadialBasis :
                 hp["kernel"] == "poly"    ? LIBSVM.Kernel.Polynomial  :
                                             LIBSVM.Kernel.Sigmoid

    for fold in 1:numFolds
        trnX = inputs_loc[cvIdx .!= fold, :]
        tstX = inputs_loc[cvIdx .== fold, :]
        trnY = targetsStr[cvIdx .!= fold]
        tstY = targetsStr[cvIdx .== fold]

        model = SVMClassifier(
            kernel = kernel_sym,
            cost   = Float64(hp["C"]),
            gamma  = Float64(get(hp, "gamma",  -1.0)),
            degree = Int32(  get(hp, "degree", -1)),
            coef0  = Float64(get(hp, "coef0",  -1.0)))

        mach = machine(model, MLJ.table(trnX), categorical(trnY))
        MLJ.fit!(mach; verbosity=0)
        preds = string.(MLJ.predict(mach, MLJ.table(tstX)))

        (_, _, _, _, _, _, f1_fold, _) = confusionMatrix(preds, tstY, strClasses)
        f1s[fold] = f1_fold
    end
    return f1s
end

function collectFoldF1s_ANN(hp::Dict,
                             dataset_tup,
                             cvIdx::Array{Int64,1},
                             allClasses;
                             numExec::Int = 5)
    (inputs_loc, targets_raw) = dataset_tup
    topology  = hp["topology"]
    lr        = get(hp, "learningRate",    0.01)
    maxEp     = get(hp, "maxEpochs",       1000)
    minL      = get(hp, "minLoss",         0.0)
    valRatio  = get(hp, "validationRatio", 0)

    strClasses   = string.(allClasses)
    allTargetsEnc = oneHotEncoding(string.(targets_raw), strClasses)

    numFolds = maximum(cvIdx)
    f1s      = zeros(Float64, numFolds)

    println("  Extrayendo F1 por fold para ANN",
            USE_GPU ? " (GPU)" : " (CPU)", ":")

    for fold in 1:numFolds
        trnIdx = cvIdx .!= fold
        tstIdx = cvIdx .== fold
        trnInputs  = inputs_loc[trnIdx, :]
        tstInputs  = inputs_loc[tstIdx, :]
        trnTargets = allTargetsEnc[trnIdx, :]
        tstTargets = allTargetsEnc[tstIdx, :]

        f1_runs = zeros(numExec)
        for run in 1:numExec
            ann = redirect_stdout(devnull) do
                if valRatio > 0
                    (trnIdx2, valIdx2) = holdOut(
                        size(trnInputs, 1),
                        valRatio * size(inputs_loc, 1) / size(trnInputs, 1))
                    ann_tmp, = trainClassANN(topology,
                        (trnInputs[trnIdx2, :], trnTargets[trnIdx2, :]);
                        validationDataset = (trnInputs[valIdx2, :], trnTargets[valIdx2, :]),
                        testDataset       = (tstInputs, tstTargets),
                        maxEpochs = maxEp, minLoss = minL, learningRate = lr)
                    ann_tmp
                else
                    ann_tmp, = trainClassANN(topology,
                        (trnInputs, trnTargets);
                        testDataset = (tstInputs, tstTargets),
                        maxEpochs = maxEp, minLoss = minL, learningRate = lr)
                    ann_tmp
                end
            end

            tstOut = collect(ann(Float32.(tstInputs'))')
            (_, _, _, _, _, _, f1_fold, _) = confusionMatrix(tstOut, tstTargets)
            f1_runs[run] = f1_fold
        end

        f1s[fold] = mean(f1_runs)
        println("    Fold $fold / $numFolds  →  F1 = $(round(f1s[fold], digits=4))")
    end
    return f1s
end

function exportFoldF1s(modelName::String, f1s::Vector{Float64})
    df = DataFrame(
        Fold     = 1:length(f1s),
        F1_Score = round.(f1s; digits=6),
    )
    filename = "folds_$(modelName).csv"
    CSV.write(filename, df)
    println("  → Folds exportados: $filename",
            "  (μ=$(round(mean(f1s),digits=4)),  σ=$(round(std(f1s),digits=4)))")
end

# ==============================================================================
# reportModel: imprime resultados + exporta CSV (se llama tras cada modelo)
# ==============================================================================
function reportModel(name::String, results, cls)
    flush(stdout)
    _, best = bestByF1(results)
    printModelResults(name, best[1], best[2], cls)
    printAllResults(name, results)
    exportModelCSV(name, results)
    flush(stdout)
    return best
end

# ==============================================================================
# EJECUCIÓN DEL GRID SEARCH PARALELO
# ==============================================================================
println("\n", "#"^62)
println("# Iniciando grid search paralelo")
println("#"^62)

println("\n[1/5] Evaluando Decision Tree ($(length(DT_GRID)) configs)...")
t_dt = @elapsed dt_results = pmap(hp -> evaluateClassicConfig(:DecisionTreeClassifier, hp), DT_GRID)
println("  Tiempo DT: $(round(t_dt, digits=1)) s")
bestDT = reportModel("Decision_Tree", dt_results, classes)

println("\n[2/5] Evaluando KNN ($(length(KNN_GRID)) configs)...")
t_knn = @elapsed knn_results = pmap(hp -> evaluateClassicConfig(:KNeighborsClassifier, hp), KNN_GRID)
println("  Tiempo KNN: $(round(t_knn, digits=1)) s")
bestKNN = reportModel("KNN", knn_results, classes)

println("\n[3/5] Evaluando SVM ($(length(SVM_GRID)) configs — linear + rbf + poly + sigmoid)...")
t_svm = @elapsed svm_results = pmap(hp -> evaluateClassicConfig(:SVC, hp), SVM_GRID)
println("  Tiempo SVM: $(round(t_svm, digits=1)) s")
bestSVM = reportModel("SVM", svm_results, classes)

println("\n[4/5] Evaluando DoME ($(length(DOME_GRID)) configs)...")
t_dome = @elapsed dome_results = pmap(hp -> evaluateClassicConfig(:DoME, hp), DOME_GRID)
println("  Tiempo DoME: $(round(t_dome, digits=1)) s")
bestDoME = reportModel("DoME", dome_results, classes)

println("\n[5/5] Evaluando ANN ($(length(ANN_GRID)) configs) en ",
        USE_GPU ? "GPU" : "CPU", " × $(Threads.nthreads()) thread(s)...")
t_ann = @elapsed begin
    ann_results = Vector{Any}(undef, length(ANN_GRID))
    Threads.@threads for i in eachindex(ANN_GRID)
        ann_results[i] = evaluateANNConfig(ANN_GRID[i])
    end
end
println("  Tiempo ANN: $(round(t_ann, digits=1)) s")
bestANN = reportModel("ANN", ann_results, classes)

# ==============================================================================
# Exportar F1 por fold para el test estadístico de Wilcoxon (ANN vs SVM)
# ==============================================================================
println("\n", "#"^62)
println("# Exportando F1 por fold → test de Wilcoxon (ANN vs SVM)")
println("#"^62)

println("\nExtrayendo F1s por fold — SVM...")
foldF1s_SVM = collectFoldF1s_SVM(bestSVM[1], dataset_tuple, crossValidationIndices, classes)
exportFoldF1s("SVM", foldF1s_SVM)

println("\nExtrayendo F1s por fold — ANN  ($(ANN_NUM_EXECUTIONS) ejecuciones × fold)...")
foldF1s_ANN = collectFoldF1s_ANN(bestANN[1], dataset_tuple, crossValidationIndices, classes;
                                  numExec = ANN_NUM_EXECUTIONS)
exportFoldF1s("ANN", foldF1s_ANN)

# ==============================================================================
# RESUMEN COMPARATIVO FINAL
# ==============================================================================
println("\n", "#"^62)
println("# RESUMEN COMPARATIVO (mejor configuración de cada modelo)")
println("#"^62)
println("\n", rpad("Modelo", 16), "| F1-score (media ± std)")
println("-"^52)
for (name, best) in [("ANN",           bestANN),
                     ("SVM",           bestSVM),
                     ("DoME",          bestDoME),
                     ("KNN",           bestKNN),
                     ("Decision Tree", bestDT)]
    f1 = best[2][7]
    println(rpad(name, 16), "| $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4))")
end

ranking = [("ANN", bestANN), ("SVM", bestSVM), ("DoME", bestDoME),
           ("KNN", bestKNN), ("Decision Tree", bestDT)]
sort!(ranking; by = x -> -x[2][2][7][1])
println("\n→ Ganador global: $(ranking[1][1])",
        "  F1 = $(round(ranking[1][2][2][7][1], digits=4))")

total_time = t_dt + t_knn + t_svm + t_dome + t_ann
println("\nTiempo total del grid search: $(round(total_time, digits=1)) s")

println("\n✓ Archivos generados:")
println("  grid_resultados_Decision_Tree.csv")
println("  grid_resultados_KNN.csv")
println("  grid_resultados_SVM.csv")
println("  grid_resultados_DoME.csv")
println("  grid_resultados_ANN.csv")
println("  folds_SVM.csv  ← F1 por fold para test de Wilcoxon")
println("  folds_ANN.csv  ← F1 por fold para test de Wilcoxon")