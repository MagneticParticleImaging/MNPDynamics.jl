using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Images
using Serialization

include("params.jl")

filenameTrain = "trainData.h5"

BTrain, pTrain = generateStructuredFields(p, tSnippet; fieldType=RANDOM_FIELD)
                     
@time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tSnippet, pTrain)

X, Y = prepareTrainData(pTrain, tSnippet, BTrain, mTrain)

inputChan = size(X,2)
outputChan = size(Y,2)

nX = normalizeData(X; dims=(1,3))
nY = normalizeData(Y; dims=(1,3))

X .= NeuralMNP.trafo(X, nX)
Y .= NeuralMNP.trafo(Y, nY)

bs = 20# 4

trainLoader = DataLoader((X[:,:,1:p[:numTrainingData]],Y[:,:,1:p[:numTrainingData]]), batchsize=bs, shuffle=true)
testLoader = DataLoader((X[:,:,(p[:numTrainingData]+1):end],Y[:,:,(p[:numTrainingData]+1):end]), batchsize=bs, shuffle=false)

modes = 12 #24
width = 32

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)

ηs = [1f-3,1f-4,1f-5]
γ = 0.5
stepSize = 30
epochs = 30

#opt = Flux.Optimiser(ExpDecay(η, γ, stepSize, 1f-5), Adam())
@time for η in ηs
  global opt = Adam(η)
  global model = NeuralMNP.train(model, opt, trainLoader, testLoader, nY; epochs, device, plotStep=1)
end

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
