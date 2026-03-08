# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(ouVPuts::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert length(ouVPuts) == length(targets) "Los vectores ouVPuts y targets deben tener la misma longitud"

    # calculamos los valores de la matriz de confusión
    VP = sum(ouVPuts .& targets)
    VN = sum(.!ouVPuts .& .!targets)
    FP = sum(ouVPuts .& .!targets)
    FN = sum(.!ouVPuts .& targets)

    # calculamos las métricas
    accuracy = (VN + VP) / (VN + VP + FN + FP)
    errorrate = (FN + FP) / (VN + VP + FN + FP)
    sensitivity = (VP + FN) == 0 ? 1.0 : VP / (VP + FN)
    specificity = (VN + FP) == 0 ? 1.0 : VN / (VN + FP)
    ppv = (VP + FP) == 0 ? 1.0 : VP / (VP + FP)
    npv = (VN + FN) == 0 ? 1.0 : VN / (VN + FN)
    f1 = (ppv + sensitivity) == 0 ? 0.0 :
        2 * (ppv * sensitivity) / (ppv + sensitivity)

    confusionmatrix = [VN FP;
                        FN VP]

    return (accuracy, errorrate, sensitivity, specificity, ppv, npv, f1, confusionmatrix)

end;

function confusionMatrix(ouVPuts::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    @assert length(ouVPuts) == length(targets) "Los vectores ouVPuts y targets deben tener la misma longitud"

    ouVPuts_bool = ouVPuts .>= threshold  # comprueba si cada uno de los valores del output es mayor que el umbral (True)
    return confusionMatrix(ouVPuts_bool, targets)

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

    metricas = confusionMatrix(outputs, targets, threshold)

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

function confusionMatrix(ouVPuts::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(ouVPuts::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(ouVPuts::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(ouVPuts::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
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


