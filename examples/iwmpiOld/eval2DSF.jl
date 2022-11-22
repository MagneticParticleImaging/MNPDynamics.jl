using MNPDynamics
using NeuralMNP
using Serialization
using Plots, StatsPlots
using Flux, NeuralOperators
using FFTW

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)

include("../visualization.jl")
include("params.jl")

function smTest(maxField, params, anisotropyAxis=nothing)

  tLengthSM = lcm(96,102);             # length of time vector
  tMaxSM = lcm(96,102) / samplingRate; # maximum evaluation time in seconds
  
  tSM = range(0, stop=tMaxSM, length=tLengthSM);
  
  fx = 2.5e6 / 102
  fy = 2.5e6 / 96
  BSM = (t, offset) -> (maxField*[sin(2*pi*fx*t), sin(2*pi*fy*t), 0] .+ offset )
  
  nOffsets = (15, 15, 1)
  
  oversampling = 1.25
  offsets = vec([ oversampling*maxField.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  if anisotropyAxis == nothing
    anisotropyAxis = vec([ 1250 .* oversampling*2.0.*(((Tuple(x)).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  else
    anisotropyAxis = vec([ Tuple(1250*anisotropyAxis)  for x in CartesianIndices(nOffsets) ])
  end
  params[:kAnis] = anisotropyAxis

  sm = simulationMNPMultiParams(BSM, tSM, offsets; params...)

  return sm
end

pSM = Dict{Symbol,Any}()
pSM[:DCore] = 20e-9        # particle diameter in nm
pSM[:kAnis] = 1250         # anisotropy constant
pSM[:derivative] = false
pSM[:neuralNetwork] = NOModel
pSM[:alg] = NeuralNetworkMNP

@time sm = smTest(0.012, pSM);
smPred = reshape(sm,:,3,15,15);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 8, 8; filename="systemMatrixPredict.svg")


α = 1.5
@time sm = smTest(0.012, pSM, [1.1*cos(pi/2*α), 1.1*sin(pi/2*α), 0.0]);
smPred = reshape(sm,:,3,15,15);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 8, 8; filename="systemMatrixPredict2.svg")