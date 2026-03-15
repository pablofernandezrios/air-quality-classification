import Pkg
Pkg.status("Flux")
Pkg.status("SymDoME")
Pkg.status("MLJ")

# Cargamos el dataset
using DelimitedFiles: readdlm
using Statistics
using Flux
using Flux.Losses
using Revise
dataset = readdlm("iris.data",',');
# Preparamos las entradas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);
targets = dataset[:,5];


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes)<=2 
        encodedfeatures = reshape(feature.==classes[1], :, 1);

    else  
        # Si hay más de 2 clases, se inicializa un vector de booleanos y se
        # itera sobre las categorías codificando según la clase que corresponda
        encodedfeatures = BitArray{2}(undef, length(feature), length(classes))
        for i in 1:length(classes)
            encodedfeatures[:, i] = feature .== classes[i]
        end
    end

    return encodedfeatures;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = (classes = unique(feature); oneHotEncoding(feature, classes))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) = outputs .>= threshold

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        return reshape(classifyOutputs(vec(outputs); threshold), :, 1)
    
    # Si hay 2 columnas, se ha representado un problema de clasificación binaria 
    # erróneamente, pero el else funcionará bien, salvo que ahora busca el máximo en vez de el mayor de 0.5
    else 
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2); 
        result = falses(size(outputs));
        result[indicesMaxEachInstance] .= true
        return result
    end 

end;

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets)

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



(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(sin.(1:8).>=0, [falses(4); trues(4)]);
@assert(isapprox(acc, 0.375) && isapprox(errorRate, 1-0.375) && isapprox(recall, 0.5) && isapprox(specificity, 0.25) && isapprox(precision, 0.4) && isapprox(NPV, 1/3.) && isapprox(F1, 4/9.) && confMatrix==[1 3; 2 2])

(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(sin.(1:8), [falses(4); trues(4)]; threshold=0.9);
@assert(isapprox(acc, 0.5) && isapprox(errorRate, 0.5) && isapprox(recall, 0.25) && isapprox(specificity, 0.75) && isapprox(precision, 0.5) && isapprox(NPV, 0.5) && isapprox(F1, 1/3.) && confMatrix==[3 1; 3 1])

(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(Bool[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1], Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]; weighted=true)
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && isapprox(F1, 1/3.) && confMatrix==[1 1 1; 1 1 1; 1 1 1])

(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(Float64[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1] .+ 1, Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]; weighted=true)
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && isapprox(F1, 1/3.) && confMatrix==[1 1 1; 1 1 1; 1 1 1])

(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(repeat(unique(targets), 50), targets)
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && isapprox(F1, 1/3.) && confMatrix==[17 17 16; 17 16 17; 16 17 17])




outputs2Classes = trainClassDoME((inputs[1:100,:], targets[1:100].=="Iris-setosa"), inputs[[1],:], 20);
@assert(isapprox(outputs2Classes[1], 1.0329783807313904));

outputs3Classes = trainClassDoME((inputs[1:149,:], oneHotEncoding(targets[1:149])), inputs[[150],:], 20);
@assert(all(isapprox.(outputs3Classes, [-1.0634799989028503 -0.4286532773153766 0.11366448575221406]; rtol=1e-3)));

outputs2Classes = trainClassDoME((inputs[1:98,:], targets[1:98]), inputs[99:100,:], 20);
@assert(all(x -> x=="Iris-versicolor", outputs2Classes));

outputs3Classes = trainClassDoME((inputs[1:149,:], targets[1:149]), inputs[[150],:], 20);
@assert(outputs3Classes[1]=="Iris-virginica");