using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization

include("params.jl")

filenameTrain = "trainData.h5"

#BTrain, pTrain = generateStructuredFields(p, tSnippet, p[:numData]; fieldType=RANDOM_FIELD)
BTrain1, pTrain1 = generateStructuredFields(p, tSnippet, p[:numData]÷2; 
        fieldType=LOWPASS_RANDOM_FIELD, freqInterval = (0.0e3, 100.0e3),
        distribution = :chi)

BTrain2, pTrain2 = generateStructuredFields(p, tSnippet, p[:numData]÷2; 
        fieldType=LOWPASS_RANDOM_FIELD, freqInterval = (0.0e3, 100.0e3), 
        anisotropyAxis = [1,0,0], dims=1,
        distribution = :chi)

#BTrain2, pTrain2 = generateStructuredFields(p, tSnippet, p[:numData]÷2; fieldType=HARMONIC_RANDOM_FIELD,
#                                            anisotropyAxis = [1,0,0], dims=1, 
#                                            freqInterval = (24.999999e3, 25.00001e3))

BTrain, pTrain = combineFields((BTrain1, BTrain2), (pTrain1, pTrain2); shuffle=true)
                     
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

modes = 12 #12#12 #24
width = 32

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
#model = NeuralMNP.make_unet_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)

#=ηs = [1f-3,1f-4]#,1f-5]
γ = 0.5
stepSize = 30
epochs = 30

@time for η in ηs
  global opt = Adam(η)
  global model = NeuralMNP.train(model, opt, trainLoader, testLoader, nY; epochs, device, plotStep=1)
end =#

η = 1f-3
γ = 0.5f0 #1f-1
stepSize = 30 #* p[:numTrainingData] / bs
epochs = 300

opt = Adam(η)
model = NeuralMNP.train(model, opt, trainLoader, testLoader, nY; epochs, device, γ, stepSize, plotStep=1)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
