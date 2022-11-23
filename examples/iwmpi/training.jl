using MNPDynamics
using NeuralMNP
using LinearAlgebra
using StatsBase, Statistics, MLUtils, Flux
using Random
using Images
using Interpolations
using Serialization

include("params.jl")

BTrain_ = rand(range(-1,1,length=1000),snippetLength,3,Z)
for z=1:Z
  for l=1:3
    filtFactor = rand(range(4,20,length=40))
    BTrain_[:,l,z] = imfilter(BTrain_[:,l,z],Kernel.gaussian((filtFactor,))) 
    BTrain_[:,l,z] ./= maximum(abs.(BTrain_[:,l,z]))
    BTrain_[:,l,z] .= maxField*(rand()*BTrain_[:,l,z]) # .+ 
                                #0.5*rand(range(-1,1,length=1000))*ones(Float32,snippetLength))
  end
end

anisotropyAxis = vec([ rand()*NeuralMNP.randAxis() for z=1:Z ]) 
p[:kAnis] =  kAnis*anisotropyAxis

filenameTrain = "trainData.h5"
@time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain_, tSnippet, p)

######

X, Y = prepareTrainData(p, tSnippet, BTrain, mTrain; useTime = false)

inputChan = size(X,2)
outputChan = size(Y,2)

nX = normalizeData(X; dims=(1,3))
nY = normalizeData(Y; dims=(1,3))

X .= NeuralMNP.trafo(X, nX)
Y .= NeuralMNP.trafo(Y, nY)

bs = 20# 4

trainLoader = DataLoader((X[:,:,1:ZTrain],Y[:,:,1:ZTrain]), batchsize=bs, shuffle=true)
testLoader = DataLoader((X[:,:,(ZTrain+1):end],Y[:,:,(ZTrain+1):end]), batchsize=bs, shuffle=false)

modes = 12 #24
width = 32

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)


ηs = [1f-3,1f-4,1f-5]
γ = 0.5
stepSize = 30
epochs = 100

#opt = Flux.Optimiser(ExpDecay(η, γ, stepSize, 1f-5), Adam())
for η in ηs
  global opt = Adam(η)
  global model = NeuralMNP.train(model, opt, trainLoader, testLoader, nY; epochs, device)
end

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, Dict{Symbol,Any}(), snippetLength)

filenameModel = "model.bin"
serialize(filenameModel, NOModel);
