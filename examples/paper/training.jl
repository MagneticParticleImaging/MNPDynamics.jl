using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization

include("params.jl")
include("generateData.jl")   

inputChan = size(XLong[1], 2)
outputChan = size(YLong[1], 2)

XTraining, YTraining = 
   generateSnippets([X[:,:,1:p[:numBaseTrainingData]] for X in XLong[1:2]], 
                    [Y[:,:,1:p[:numBaseTrainingData]] for Y in YLong[1:2]], 
                     p[:numTrainingData], [0.8,0.2], p[:snippetLength])

nX = normalizeData(XTraining; dims=(1,3))
nY = normalizeData(YTraining; dims=(1,3))
XTraining .= NeuralMNP.trafo(XTraining, nX)
YTraining .= NeuralMNP.trafo(YTraining, nY)

bs = 20# 4
trainLoader = DataLoader((XTraining, YTraining), batchsize=bs, shuffle=true)

# gen validation data
validationLoaders = Any[]
for l = 3:4
  R = (p[:numBaseTrainingData]+1):(p[:numBaseTrainingData]+p[:numBaseValidationData])
  XVal, YVal = generateSnippets([XLong[l][:,:,R]], [YLong[l][:,:,R]], 
                     p[:numValidationData], [1.0], p[:snippetLength])
  XVal .= NeuralMNP.trafo(XVal, nX)
  YVal .= NeuralMNP.trafo(YVal, nY)
  push!(validationLoaders, DataLoader((XVal,YVal), batchsize=bs, shuffle=false))
end


modes = 16 #18 #24
width = 32 #48

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
#model = NeuralMNP.make_unet_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)


η = 1f-3
γ = 0.5f0 #1f-1
stepSize = 20   #* p[:numTrainingData] / bs
epochs = 100

opt = Adam(η)

model = NeuralMNP.train(model, opt, trainLoader, validationLoaders, nY; 
                        epochs, device, γ, stepSize, plotStep=1, logging=true)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
