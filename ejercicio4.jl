# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    vp = count(outputs .& targets)
    vn = count(.!outputs .& .!targets)
    fp = count(outputs .& .!targets)
    fn = count(.!outputs .& targets)
    confusionmatrix = [[vn fp]; [fn vp]]

    acc = (vn + vp) / (vn + vp + fn + fp)    
    errorRate = (fn + fp) / (vn + vp + fn + fp)  # Sería equivalente a (1 - acc)
    recall = (vp + fn) == 0 ? 1.0 : (vp) / (vp + fn)
    specificity = (vn + fp) == 0 ? 1.0 : (vn) / (vn + fp)
    precision = (vp + fp) == 0 ? 1.0 : (vp) / (vp + fp)
    NPV = (vn + fn) == 0 ? 1.0 : (vn) / (vn + fn)
    f1 = (recall + precision) == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall)

    return (acc=acc, errorRate=errorRate, recall=recall, specificity=specificity, 
            precision=precision, NPV=NPV, f1=f1, confusionMatrix=confusionmatrix)
end;


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return confusionMatrix(outputs, targets)
end;


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    println(confusionMatrix(outputs, targets))
end;


function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    print(confusionMatrix(outputs, targets, threshold=threshold))
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "outputs y targets deben tener el mismo tamaño"

    numClasses = size(outputs, 2)

    @assert numClasses != 2 "Para problemas binarios usa la versión unidimensional"

    if numClasses == 1  
        confusionMatrix(outputs, targets)
    end

    recall = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    NPV = zeros(numClasses)
    f1 = zeros(numClasses)

    for i in 1:numClasses
        metrics = confusionMatrix(outputs[:, i], targets[:, i])
        recall[i] = metrics.recall
        specificity[i] = metrics.specificity
        precision[i] = metrics.precision
        NPV[i] = metrics.NPV
        f1[i] = metrics.f1
    end
    confusionmatrix = targets' * outputs

    instances = vec(sum(targets, dims=1))
    if weighted
        recall = sum(recall .* instances) / (sum(instances))
        specificity = sum(specificity .* instances) / (sum(instances))
        precision = sum(precision .* instances) / (sum(instances))
        NPV = sum(NPV .* instances) / (sum(instances))
        f1 = sum(f1 .* instances) / (sum(instances))
    else    
        recall = sum(recall) / numClasses
        specificity = sum(specificity) / numClasses
        precision = sum(precision) / numClasses
        NPV = sum(NPV) / numClasses
        f1 = sum(f1) / numClasses
    end

    # Calculamos el valor de precisión con la función accuracy desarrollada en una práctica anterior
    acc = accuracy(outputs, targets)
    errorRate = 1 - acc
    
    return (acc=acc, errorRate=errorRate, recall=recall, specificity=specificity, 
            precision=precision, NPV=NPV, f1=f1, confusionMatrix=confusionmatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputs = classifyOutputs(outputs, threshold=threshold)
    return confusionMatrix(outputs, targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "outputs y targets deben tener el mismo tamaño"

    @assert all([in(label, classes) for label in unique(vcat(targets, outputs))]) "Las etiquetas deben estar incluidas en classes"

    outputs = oneHotEncoding(outputs, classes)
    targets = oneHotEncoding(targets, classes)

    return confusionMatrix(outputs, targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    println(confusionMatrix(outputs, targets, classes; weighted=weighted))
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    println(confusionMatrix(outputs, targets; weighted=weighted))
end;



using SymDoME
using GeneticProgramming

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64, 2}, trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = convert(AbstractArray{Float64, 2}, testInputs)

    model, _, _, _ = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes) 
    model = string(model)

    testOutputs = evaluateTree(model, testInputs)
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end

    return testOutputs
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    numClasses = size(trainingTargets, 2)

    if numClasses == 1
        trainingTargets = vec(trainingDataset[2])
        outputs = trainClassDoME((trainingInputs, trainingTargets), testInputs, maximumNodes)
        return reshape(outputs, :, 1)
    end
    
    numSamples = size(testInputs, 1)
    allOutputs = Array{Float64,2}(undef, numSamples, numClasses)

    for i in 1:numClasses
        classTargets = trainingTargets[:, i] 
        outputs = trainClassDoME((trainingInputs, classTargets), testInputs, maximumNodes)
        allOutputs[:, i] = outputs
    end

    return allOutputs
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    classes = unique(trainingTargets)  # Se calcula el número de clases en el objetivo
    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs,1))

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes)

    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool); 
        testOutputs[ testOutputsBool] .= classes[1]; 
        if length(classes)==2 
            testOutputs[.!testOutputsBool] .= classes[2]; 
        end
    else
        for numClass in eachindex(classes)
            testOutputs[testOutputsBool[:,numClass]] .= classes[numClass]
        end
    end

    return testOutputs
end
