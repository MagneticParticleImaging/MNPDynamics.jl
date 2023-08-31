using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization

include("params.jl")
include("generateData.jl")   

inputChan = size(X1,2)
outputChan = size(Y1,2)

nX = normalizeData(X1; dims=(1,3))
nY = normalizeData(Y1; dims=(1,3))

X1 .= NeuralMNP.trafo(X1, nX)
Y1 .= NeuralMNP.trafo(Y1, nY)
X2 .= NeuralMNP.trafo(X2, nX)
Y2 .= NeuralMNP.trafo(Y2, nY)


bs = 20# 4

XTraining1 = X1[:,:,1:p[:numTrainingData]]
YTraining1 = Y1[:,:,1:p[:numTrainingData]]
trainLoader1 = DataLoader((XTraining1, YTraining1), batchsize=bs, shuffle=true)

XTraining2 = cat(X1[:,:,1:p[:numTrainingData]], 
                X2[:,:,1:(p[:numTrainingData]÷10)], dims=3)
                #X3[:,:,1:p[:numTrainingData]],
                #X4[:,:,1:p[:numTrainingData]], dims=3)
YTraining2 = cat(Y1[:,:,1:p[:numTrainingData]], 
                Y2[:,:,1:(p[:numTrainingData]÷10)], dims=3)
                #Y3[:,:,1:p[:numTrainingData]],
                #Y4[:,:,1:p[:numTrainingData]], dims=3)

trainLoader2 = DataLoader((XTraining2, YTraining2), batchsize=bs, shuffle=true)


testLoaders = Any[]
push!(testLoaders, DataLoader((X1[:,:,(p[:numTrainingData]+1):end],
             Y1[:,:,(p[:numTrainingData]+1):end]), batchsize=bs, shuffle=false))
push!(testLoaders, DataLoader((X2[:,:,(p[:numTrainingData]+1):end],
             Y2[:,:,(p[:numTrainingData]+1):end]), batchsize=bs, shuffle=false))

modes = 18 #24
width = 48

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
#model = NeuralMNP.make_unet_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)


η = 1f-3
γ = 0.5f0 #1f-1
stepSize = 10   #* p[:numTrainingData] / bs
epochs = 32

opt = Adam(η)
#model = NeuralMNP.train(model, opt, trainLoader1, testLoaders, nY; 
#                        epochs, device, γ, stepSize, plotStep=1)

#opt = Adam(η)
model = NeuralMNP.train(model, opt, trainLoader2, testLoaders, nY; 
                        epochs, device, γ, stepSize, plotStep=1)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
