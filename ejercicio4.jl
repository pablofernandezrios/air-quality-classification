# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    Matriz_confusion = [TN FP; FN TP]  # ← orden corregido

    accuracy    = (TP + TN) / length(outputs)
    tasa_fallo  = 1 - accuracy
    sensibilidad              = (TP + FN) == 0 ? 1.0 : TP / (TP + FN)
    especifidad               = (TN + FP) == 0 ? 1.0 : TN / (TN + FP)
    valor_predictivo_positivo = (TP + FP) == 0 ? 1.0 : TP / (TP + FP)
    valor_predictivo_negativo = (TN + FN) == 0 ? 1.0 : TN / (TN + FN)
    F1_score = (valor_predictivo_positivo + sensibilidad) == 0 ? 0.0 :
        2 * (valor_predictivo_positivo * sensibilidad) / (valor_predictivo_positivo + sensibilidad)

    return accuracy, tasa_fallo, sensibilidad, especifidad,
           valor_predictivo_positivo, valor_predictivo_negativo, F1_score, Matriz_confusion
end;

# 1. Real outputs → Bool via threshold, then call Bool version
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    confusionMatrix(outputs .>= threshold, targets)
end; 

# 2. Multiclass Bool 2D: one-hot outputs and targets
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "outputs and targets must have the same size"
    numClasses = size(outputs, 2)

    # If binary encoded as 2 columns, delegate to 1D
    if numClasses == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end

    # Per-class metrics (one-vs-rest)
    metrics = [confusionMatrix(outputs[:,i], targets[:,i]) for i in 1:numClasses]

    # Unpack: each element is (acc, fail, sens, spec, vpp, vpn, f1, matrix)
    sensibilidades = [m[3] for m in metrics]
    especifidades  = [m[4] for m in metrics]
    vpps           = [m[5] for m in metrics]
    vpns           = [m[6] for m in metrics]
    f1s            = [m[7] for m in metrics]

    # Class weights based on number of positive samples in targets
    classWeights = vec(sum(targets, dims=1)) ./ size(targets, 1)

    if weighted
        sensibilidad              = sum(sensibilidades .* classWeights)
        especifidad               = sum(especifidades  .* classWeights)
        valor_predictivo_positivo = sum(vpps           .* classWeights)
        valor_predictivo_negativo = sum(vpns           .* classWeights)
        F1_score                  = sum(f1s            .* classWeights)
    else
        sensibilidad              = mean(sensibilidades)
        especifidad               = mean(especifidades)
        valor_predictivo_positivo = mean(vpps)
        valor_predictivo_negativo = mean(vpns)
        F1_score                  = mean(f1s)
    end

    # Overall accuracy: fraction of fully correct rows
    accuracy   = sum(all(outputs .== targets, dims=2)) / size(targets, 1)
    tasa_fallo = 1 - accuracy

    # Full confusion matrix (numClasses × numClasses)
    # Row = true class, Col = predicted class
    Matriz_confusion = zeros(Int, numClasses, numClasses)
    for i in 1:size(targets, 1)
        true_class = findfirst(targets[i,:])
        pred_class = findfirst(outputs[i,:])
        if !isnothing(true_class) && !isnothing(pred_class)
            Matriz_confusion[true_class, pred_class] += 1
        end
    end

    return accuracy, tasa_fallo, sensibilidad, especifidad,
           valor_predictivo_positivo, valor_predictivo_negativo, F1_score, Matriz_confusion
end;

# 4. Categorical 1D with explicit classes list
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1},
                         classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert length(outputs) == length(targets) "outputs and targets must have the same length"

    numClasses = length(classes)

    # Encode as one-hot Bool matrices
    function oneHot(v, classes)
        m = falses(length(v), length(classes))
        for (i, val) in enumerate(v)
            idx = findfirst(==(val), classes)
            if !isnothing(idx)
                m[i, idx] = true
            end
        end
        return m
    end

    outputs_oh = oneHot(outputs, classes)
    targets_oh = oneHot(targets, classes)

    confusionMatrix(outputs_oh, targets_oh; weighted=weighted)
end;

# 5. Categorical 1D without explicit classes: infer from targets
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = sort(unique(targets))  # orden estable
    confusionMatrix(outputs, targets, classes; weighted=weighted)
end;
# Fix 1: Real 2D → usar argmax en lugar de threshold elemento a elemento
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                         threshold::Real=0.5, weighted::Bool=true)
    bool_outputs = falses(size(outputs))
    for i in 1:size(outputs, 1)
        bool_outputs[i, argmax(outputs[i,:])] = true
    end
    confusionMatrix(bool_outputs, targets; weighted=weighted)
end;


using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;
