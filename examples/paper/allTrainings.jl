using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization
using DataFrames

include("params.jl")
include("generateData.jl")   

for l=1:size(dfTraining, 1)
  # load training data
  dsTraining = dfTraining.datasetTraining[l]
  dsTrainingWeighting = dfTraining.datasetTrainingWeighting[l]
  numTrainingData = dfTraining.numTrainingData[l]

  inputChan = size(XLong[dsTraining[1]], 2)
  outputChan = size(YLong[dsTraining[1]], 2)


  XTraining, YTraining = 
    generateSnippets([X[:,:,1:p[:numBaseTrainingData]] for X in XLong[dsTraining]], 
                      [Y[:,:,1:p[:numBaseTrainingData]] for Y in YLong[dsTraining]], 
                      numTrainingData, dsTrainingWeighting, p[:snippetLength])

  nX = normalizeData(XTraining; dims=(1,3))
  nY = normalizeData(YTraining; dims=(1,3))
  XTraining .= NeuralMNP.trafo(XTraining, nX)
  YTraining .= NeuralMNP.trafo(YTraining, nY)

  bs = dfTraining.batchSize[l]
  trainLoader = DataLoader((XTraining, YTraining), batchsize=bs, shuffle=true)
  
  # load validation data

  dsValidation = dfTraining.datasetValidation[l]
  validationLoaders = Any[]
  for k = 1:length(dsValidation)
    R = (p[:numBaseTrainingData]+1):(p[:numBaseTrainingData]+p[:numBaseValidationData])
    XVal, YVal = generateSnippets([XLong[dsValidation[k]][:,:,R]], [YLong[dsValidation[k]][:,:,R]], 
                      p[:numValidationData], [1.0], p[:snippetLength])
    XVal .= NeuralMNP.trafo(XVal, nX)
    YVal .= NeuralMNP.trafo(YVal, nY)
    push!(validationLoaders, DataLoader((XVal,YVal), batchsize=bs, shuffle=false))
  end
  
  # build network

  modes = dfTraining.networkModes[l]
  width = dfTraining.networkWidth[l]
  
  model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)
  
  # do training

  η = dfTraining.η[l]
  γ = dfTraining.γ[l]
  stepSize = dfTraining.stepSize[l]
  epochs = dfTraining.epochs[l]

  opt = Adam(η)

  model = NeuralMNP.train(model, opt, trainLoader, validationLoaders, nY; 
                          epochs, device, γ, stepSize, plotStep=-1, logging=true)

  NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, p, p[:snippetLength])

  filenameModel = joinpath(modeldir, "model.bin")
  serialize(filenameModel, NOModel);
end



