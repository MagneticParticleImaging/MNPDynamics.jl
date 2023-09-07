using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization
using DataFrames

include("params.jl")
include("generateData.jl")   

X1 = XLong[1]
Y1 = YLong[1]
X2 = XLong[2]
Y2 = YLong[2]

inputChan = size(X1,2)
outputChan = size(Y1,2)


bs = 20# 4

XTraining2 = cat(X1[:,:,1:p[:numTrainingData]], 
                X2[:,:,1:(p[:numTrainingData]÷10)], dims=3)

YTraining2 = cat(Y1[:,:,1:p[:numTrainingData]], 
                Y2[:,:,1:(p[:numTrainingData]÷10)], dims=3)


nX = normalizeData(X1; dims=(1,3))
nY = normalizeData(Y1; dims=(1,3))

X1 .= NeuralMNP.trafo(X1, nX)
Y1 .= NeuralMNP.trafo(Y1, nY)
X2 .= NeuralMNP.trafo(X2, nX)
Y2 .= NeuralMNP.trafo(Y2, nY)

trainLoader2 = DataLoader((XTraining2, YTraining2), batchsize=bs, shuffle=true)




testLoaders = Any[]
push!(testLoaders, DataLoader((X1[:,:,(p[:numTrainingData]+1):end],
             Y1[:,:,(p[:numTrainingData]+1):end]), batchsize=bs, shuffle=false))
push!(testLoaders, DataLoader((X2[:,:,(p[:numTrainingData]+1):end],
             Y2[:,:,(p[:numTrainingData]+1):end]), batchsize=bs, shuffle=false))

modes = 12 #24
width = 32

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
#model = NeuralMNP.make_unet_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)


η = 1f-3
γ = 0.5f0 #1f-1
stepSize = 10   #* p[:numTrainingData] / bs
epochs = 20

opt = Adam(η)
#model = NeuralMNP.train(model, opt, trainLoader1, testLoaders, nY; 
#                        epochs, device, γ, stepSize, plotStep=1)

#opt = Adam(η)
model = NeuralMNP.train(model, opt, trainLoader2, testLoaders, nY; 
                        epochs, device, γ, stepSize, plotStep=1)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
