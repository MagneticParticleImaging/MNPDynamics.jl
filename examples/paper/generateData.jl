using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization
using DataFrames

include("params.jl")

XLong = Any[]
YLong = Any[]

for l=1:size(dfDatasets, 1)
  filenameTrain = joinpath(datadir, dfDatasets.filename[l])
  BTrain = generateRandomFields(tBaseData, dfDatasets.numData[l]; 
                                      fieldType = dfDatasets.fieldType[l], 
                                      dims = dfDatasets.fieldDims[l],
                                      filterFactor = dfDatasets.filterFactor[l],
                                      maxField = dfDatasets.maxField[l])

  pTrain = generateRandomParticleParams(p, dfDatasets.numData[l]; 
                                        anisotropyAxis = dfDatasets.anisotropyAxis[l],
                                        distribution = dfDatasets.samplingDistribution[l])

  @time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tBaseData, pTrain)


  X, Y = prepareTrainData(pTrain, tBaseData, BTrain, mTrain)
  push!(XLong, X)
  push!(YLong, Y)
end



function generateSnippets(Xs, Ys, numData, weights, snippetLength)
  weights ./= sum(weights)
  N = length(Xs)
  numDataEachSet = zeros(Int, N)
  counter = numData
  for l = 1:(N-1)
    numDataEachSet[l] = round(Int, numData*weights[l])
    counter -= numDataEachSet[l]
  end
  numDataEachSet[end] = counter

  @assert sum(numDataEachSet) == numData

  XOut = zeros(eltype(Xs[1]), snippetLength, size(Xs[1],2), numData)
  YOut = zeros(eltype(Ys[1]), snippetLength, size(Ys[1],2), numData)
  
  counter = 1
  for l = 1:N
    M = size(Xs[l],3)
    numSnippetEachConfiguration = ceil(Int, numDataEachSet[l] / M)
    currConfig = 1
    for j = 1:numDataEachSet[l]
      currOffset = max(size(Xs[l],1) - snippetLength -
                   floor(Int, (size(Xs[l],1) - snippetLength) * ((j-1)÷M) / numSnippetEachConfiguration), 1)
      
      XOut[:,:,counter] .= Xs[l][currOffset:currOffset+snippetLength-1,:,currConfig]
      YOut[:,:,counter] .= Ys[l][currOffset:currOffset+snippetLength-1,:,currConfig]
      currConfig = mod1(currConfig+1, M)
      counter += 1
    end
  end
  return XOut, YOut
end