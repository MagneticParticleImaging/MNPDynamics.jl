using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization
using DataFrames

include("params.jl")

Random.seed!(seed)

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

  @time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tBaseData, pTrain; force=forceDataGen)

  X, Y = prepareTrainData(pTrain, tBaseData, BTrain, mTrain)
  push!(XLong, X)
  push!(YLong, Y)
end


Random.seed!(seed)
