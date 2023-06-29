using MNPDynamics
using NeuralMNP
using Plots, Plots.Measures
using FFTW, HDF5
using Serialization
using Flux
using Statistics


include("../visualization.jl")
include("utils.jl")

function calcSMs(p; device=gpu)

  sm = Dict{Symbol,Any}()

  p[:anisotropyAxis] = nothing
  @time sm[:FluidFNO] = calcSM(p; device)
  α = 1.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized135FNO] = calcSM(p; device)
  α = 0.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized45FNO] = calcSM(p; device)

 #= delete!(p, :neuralNetwork)
  delete!(p, :alg)

  p[:anisotropyAxis] = nothing
  @time sm[:FluidFokkerPlanck] = calcSM(p; device)
  α = 1.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized135FokkerPlanck] = calcSM(p; device)
  α = 0.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized45FokkerPlanck] = calcSM(p; device) =#

  return sm
end

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
pSM = copy(NOModel.params)
pSM[:DCore] = 20e-9        # particle diameter in nm
pSM[:kAnis] = 1250         # anisotropy constant
pSM[:derivative] = false
pSM[:neuralNetwork] = NOModel
pSM[:alg] = NeuralNetworkMNP
N = 30
pSM[:nOffsets] = (N, N, 1)
pSM[:maxField] = 0.012
pSM[:dividers] = (102,96,1)

filenameSMs = "sm.bin"
if false #isfile(filenameSMs)
  sm = deserialize(filenameSMs)
else
  sm = calcSMs(pSM, device=gpu)
  serialize(filenameSMs, sm)
end

MX = MY = 4

plot2DSM(rfft(reshape(sm[:FluidFNO],:,3,N,N),1), MX, MY; filename="systemMatrixFluidFNO.png")
plot2DSM(rfft(reshape(sm[:Immobilized135FNO],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized135FNO.png")
plot2DSM(rfft(reshape(sm[:Immobilized45FNO],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized45FNO.png")
#=plot2DSM(rfft(reshape(sm[:FluidFokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixFluidFokkerPlanck.png")
plot2DSM(rfft(reshape(sm[:Immobilized135FokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized135FokkerPlanck.png")
plot2DSM(rfft(reshape(sm[:Immobilized45FokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized45FokkerPlanck.png")

@info relError(sm[:FluidFNO], sm[:FluidFokkerPlanck])
@info relError(sm[:Immobilized45FNO], sm[:Immobilized45FokkerPlanck])
@info relError(sm[:Immobilized135FNO], sm[:Immobilized135FokkerPlanck])=#
