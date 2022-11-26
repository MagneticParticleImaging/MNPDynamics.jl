using MNPDynamics
using NeuralMNP
using Plots, Measures
using FFTW, HDF5
using ProgressMeter
@everywhere using StaticArrays

include("../visualization.jl")
include("utils.jl")

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
pSM = copy(NOModel.params)
pSM[:DCore] = 20e-9        # particle diameter in nm
pSM[:kAnis] = 1250         # anisotropy constant
pSM[:derivative] = false
pSM[:neuralNetwork] = NOModel
pSM[:alg] = NeuralNetworkMNP
N = 10
pSM[:nOffsets] = (N, N, N)
pSM[:maxField] = 0.012
pSM[:dividers] = (102,96,99)

pSM[:anisotropyAxis] = nothing
@time sm = calcSM(pSM; device=cpu);
