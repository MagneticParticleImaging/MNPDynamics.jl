using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization
using DataFrames

include("params.jl")
include("generateData.jl")   


inputChan = size(XLong[1], 2)
outputChan = size(YLong[1], 2)

XTraining, YTraining = 
   generateSnippets([X[:,:,1:p[:numBaseTrainingData]] for X in XLong[1:2]], 
                    [Y[:,:,1:p[:numBaseTrainingData]] for Y in YLong[1:2]], 
                     p[:numTrainingData], [0.9,0.1], p[:snippetLength])

nX = normalizeData(XTraining; dims=(1,3))
nY = normalizeData(YTraining; dims=(1,3))
XTraining .= NeuralMNP.trafo(XTraining, nX)
YTraining .= NeuralMNP.trafo(YTraining, nY)

bs = 20# 4
trainLoader = DataLoader((XTraining, YTraining), batchsize=bs, shuffle=true)

# gen validation data
validationLoaders = Any[]
for l = 1:2
  R = (p[:numBaseTrainingData]+1):(p[:numBaseTrainingData]+p[:numBaseValidationData])
  XVal, YVal = generateSnippets([XLong[l][:,:,R]], [YLong[l][:,:,R]], 
                     p[:numValidationData], [1.0], p[:snippetLength])
  XVal .= NeuralMNP.trafo(XVal, nX)
  YVal .= NeuralMNP.trafo(YVal, nY)
  push!(validationLoaders, DataLoader((XVal,YVal), batchsize=bs, shuffle=false))
end

# gen validation data
testLoaders = Any[]
for l = 1:2
  R = (p[:numBaseTrainingData]+p[:numBaseValidationData]+1):(p[:numBaseTrainingData]+p[:numBaseValidationData]+p[:numBaseTestData])
  XVal, YVal = generateSnippets([XLong[l][:,:,R]], [YLong[l][:,:,R]], 
                     p[:numTestData], [1.0], p[:snippetLength])
  XVal .= NeuralMNP.trafo(XVal, nX)
  YVal .= NeuralMNP.trafo(YVal, nY)
  push!(testLoaders, DataLoader((XVal,YVal), batchsize=bs, shuffle=false))
end


modes = 18 #24
width = 48

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
#model = NeuralMNP.make_unet_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)


η = 1f-3
γ = 0.5f0 #1f-1
#γ = 1f-1
stepSize = 30   #* p[:numTrainingData] / bs
epochs = 300

opt = Adam(η)

model = NeuralMNP.train(model, opt, trainLoader, validationLoaders, nY; 
                        epochs, device, γ, stepSize, plotStep=1, logging=true)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
#filenameModel = "model.bin"
#NOModel = deserialize(filenameModel)


function plotErrorStatistics(p, neuralNetwork, X, Y)
  ND = 10
  NK = 5
  intervalD = range(p[:DCore][1], p[:DCore][2], length=ND)
  intervalK = range(p[:kAnis][1], p[:kAnis][2], length=NK)

  numSignals = size(X,3)

  error = zeros(ND, NK)
  errorCount = ones(ND, NK)

  X_ = back(X, neuralNetwork.normalizationX)
  Y_ = back(Y, neuralNetwork.normalizationY)

  for z = 1:numSignals
    kAnis = norm(X_[1,5:7,z])
    D = X_[1,4,z]

    xc = X[:,:,z:z]
    yc = back(neuralNetwork.model(Float32.(xc)), neuralNetwork.normalizationY)
    err = norm(yc - Y_[:,:,z:z]) / norm(Y_[:,:,z:z])

    d = round(Int, (D-p[:DCore][1]) / (p[:DCore][2]-p[:DCore][1]) * (ND-1) + 1)
    k = round(Int, (kAnis-p[:kAnis][1]) / (p[:kAnis][2]-p[:kAnis][1]) * (NK-1) + 1)

    #@info typeof(err)
    error[d,k] += err
    errorCount[d,k] += 1
  end

  error = error ./ errorCount

  p_ = heatmap(error', c=:viridis, xlabel="DCore", ylabel="kAnis", xticks=(1:ND,round.(collect(intervalD).*1e9, digits=2)),
  yticks=(1:NK,round.(collect(intervalK), digits=0)))

  savefig(p_, "img/statisticalError.pdf")
  p_
end

plotErrorStatistics(p, NOModel, validationLoaders[1].data[1], validationLoaders[1].data[2])
