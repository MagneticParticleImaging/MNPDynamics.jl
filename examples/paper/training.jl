@time using MNPDynamics
using Plots, Measures
using FFTW
using Flux, MLUtils
using BSON

include("params.jl")

filenameTrain = "trainData.h5"

#BTrain, pTrain = generateStructuredFields(p, tSnippet, Z; maxField=maxField, 
#                     fieldType=RANDOM_FIELD, filterFactor=20)
BTrain, pTrain = generateStructuredFields(p, tSnippet, Z; maxField=maxField, 
                     fieldType=HARMONIC_RANDOM_FIELD, filterFactor=20)
                     
mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tSnippet, pTrain)

X, Y = prepareTrainData(pTrain, tSnippet, BTrain, mTrain; useTime = true)

inputChan = size(X,2)
outputChan = size(Y,2)

nX = normalizeData(X; dims=(1,3))
nY = normalizeData(Y; dims=(1,3))

X .= trafo(X, nX)
Y .= trafo(Y, nY)


bs = 20# 4


trainLoader = DataLoader((X[:,:,1:ZTrain],Y[:,:,1:ZTrain]), batchsize=bs, shuffle=true)
testLoader = DataLoader((X[:,:,(ZTrain+1):end],Y[:,:,(ZTrain+1):end]), batchsize=bs, shuffle=false)

modes = 8 #24
width = 32
model = make_neural_operator_model(inputChan, outputChan, modes, width, MNPDynamics.NeuralOperators.FourierTransform)

η = 1f-2
γ = 0.5
stepSize = 100
#opt = Flux.Optimiser(ExpDecay(η, γ, stepSize, 1f-6), Adam())
opt = Adam(η)

MNPDynamics.train(model, opt, trainLoader, testLoader, nY; epochs=30)

NOModel = NeuralNetwork(model, nX, nY, p, snippetLength)

filenameModel = "model.bson"
bson(filenameModel, model = NOModel);


#=
filenameTest = "testData.h5"
BTest, pTest = generateStructuredFields(p, tSnippet, ZTest; maxField=maxField, fieldType=HARMONIC_RANDOM_FIELD)
mTest, BTest = simulationMNPMultiParams(filenameTest, BTest, tSnippet, pTest)
XTest, YTest = prepareTrainData(pTest, tSnippet, BTest, mTest; useTime = true)
XTest .= trafo(XTest, nX)
YTest .= trafo(YTest, nY)
testLoader2 = DataLoader((XTest, YTest), batchsize=bs, shuffle=true)

MNPDynamics.train(model, opt, trainLoader, testLoader2, nY; epochs=3)
=#



