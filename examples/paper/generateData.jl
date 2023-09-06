using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics, MLUtils, Flux
using Random
using Serialization

include("params.jl")

Random.seed!(1)

filenameTrain1 = joinpath(datadir, "trainData1.h5")
#BTrain1, pTrain1 = generateStructuredFields(p, tSnippet, p[:numData]; 
#                                      fieldType=RANDOM_FIELD)

BTrain1 = generateRandomFields(tSnippet, p[:numData]; 
                                      fieldType = RANDOM_FIELD, 
                                      dims = 1:3,
                                      filterFactor = p[:filterFactor],
                                      maxField = p[:maxField])

pTrain1 = generateRandomParticleParams(p, p[:numData]; 
                                        anisotropyAxis = nothing,
                                        distribution = :uniform)

@time mTrain1, BTrain1 = simulationMNPMultiParams(filenameTrain1, BTrain1, tSnippet, pTrain1, force=true)
X1, Y1 = prepareTrainData(pTrain1, tSnippet, BTrain1, mTrain1)

filenameTrain2 = joinpath(datadir, "trainData2.h5")
#BTrain2, pTrain2 = generateStructuredFields(p, tSnippet, p[:numData]; 
#                                      fieldType=RANDOM_FIELD, 
#                                      anisotropyAxis = [1,0,0], dims=1)
BTrain2 = generateRandomFields(tSnippet, p[:numData]; 
                                      fieldType = RANDOM_FIELD, 
                                      dims = 1,
                                      filterFactor = p[:filterFactor],
                                      maxField = p[:maxField])

pTrain2 = generateRandomParticleParams(p, p[:numData]; 
                                        anisotropyAxis = [1,0,0],
                                        distribution = :uniform)

@time mTrain2, BTrain2 = simulationMNPMultiParams(filenameTrain2, BTrain2, tSnippet, pTrain2, force=true)
X2, Y2 = prepareTrainData(pTrain2, tSnippet, BTrain2, mTrain2)

Random.seed!(1)

#=
filenameTrain3 = joinpath(datadir, "trainData3.h5")
BTrain3, pTrain3 = generateStructuredFields(p, tSnippet, p[:numData]; 
                                      fieldType=HARMONIC_RANDOM_FIELD, 
                                      freqInterval = (10e3, 50e3))
@time mTrain3, BTrain3 = simulationMNPMultiParams(filenameTrain3, BTrain3, tSnippet, pTrain3)
X3, Y3 = prepareTrainData(pTrain3, tSnippet, BTrain3, mTrain3)

filenameTrain4 = joinpath(datadir, "trainData4.h5")
BTrain4, pTrain4 = generateStructuredFields(p, tSnippet, p[:numData]; 
                                      fieldType=HARMONIC_RANDOM_FIELD, 
                                      anisotropyAxis = [1,0,0], dims=1,
                                      freqInterval = (10e3, 50e3))
@time mTrain4, BTrain4 = simulationMNPMultiParams(filenameTrain4, BTrain4, tSnippet, pTrain4)
X4, Y4 = prepareTrainData(pTrain4, tSnippet, BTrain4, mTrain4)
=#