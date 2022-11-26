using MNPDynamics
using NeuralMNP
using Plots, Measures
using FFTW, HDF5
using Serialization
using Flux
using Statistics
using StaticArrays

include("params.jl")
include("../visualization.jl")


function smTest(params, maxField, samplingRate, anisotropyAxis=nothing; device=cpu)

  tLengthSM = lcm(96,102);             # length of time vector
  tMaxSM = lcm(96,102) / samplingRate; # maximum evaluation time in seconds
  
  tSM = range(0, step=1/samplingRate, length=tLengthSM);
  
  fx = 2.5e6 / 102
  fy = 2.5e6 / 96
  BSM = (t, offset) -> SVector{3,Float32}(maxField*sin(2*pi*fx*t)+offset[1], 
                                          maxField*sin(2*pi*fy*t)+offset[2], offset[3])
  
  nOffsets = (30, 30, 1)
    
  factor = 1250.0
  
  oversampling = 1.25
  offsets = vec([ oversampling*maxField.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  if anisotropyAxis == nothing
    anisotropyAxis = vec([ factor.*oversampling*2.0.*(((Tuple(x)).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  else
    anisotropyAxis = vec([ Tuple(factor.*anisotropyAxis)  for x in CartesianIndices(nOffsets) ])
  end
  params[:kAnis] = anisotropyAxis


  if haskey(params, :neuralNetwork) && params[:neuralNetwork] != nothing

    sm = zeros(Float32, tLengthSM,  3, prod(nOffsets))

    for z=1:prod(nOffsets)

      off = offsets[z]
      Bx = [BSM(t_, off)[1] for t_ in tSM]
      By = [BSM(t_, off)[2] for t_ in tSM]
      Bz = [BSM(t_, off)[3] for t_ in tSM]
      n_ = anisotropyAxis[z]


      XLongTest = cat( Bx, By, Bz, n_[1]*ones(Float32, tLengthSM), n_[2]*ones(Float32, tLengthSM),
                      n_[3]*ones(Float32, tLengthSM), dims=2)

      q = NeuralMNP.applyToArbitrarySignal(params[:neuralNetwork], XLongTest, device)
      sm[:,:,z] .= q
    end
    return sm
  else
    sm = simulationMNPMultiParams(BSM, tSM, offsets; params...)
    return sm  
  end
  
end

function calcSMs(p)

  maxField = 0.012

  sm = Dict{Symbol,Any}()

  @time sm[:FluidFNO] = smTest(p, maxField, samplingRate);
  α = 1.5
  @time sm[:Immobilized135FNO] = smTest(p, maxField, samplingRate, (cos(pi/2*α), sin(pi/2*α), 0.0));
  α = 0.5
  @time sm[:Immobilized45FNO] = smTest(p, maxField, samplingRate, (cos(pi/2*α), sin(pi/2*α), 0.0));

  delete!(p, :neuralNetwork)
  @time sm[:FluidFokkerPlanck] = smTest(p, maxField, samplingRate);
  α = 1.5
  @time sm[:Immobilized135FokkerPlanck] = smTest(p, maxField, samplingRate, (cos(pi/2*α), sin(pi/2*α), 0.0));
  α = 0.5
  @time sm[:Immobilized45FokkerPlanck] = smTest(p, maxField, samplingRate, (cos(pi/2*α), sin(pi/2*α), 0.0));

  return sm
end

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
p[:neuralNetwork] = NOModel

filenameSMs = "sm.bin"
if isfile(filenameSMs)
  sm = deserialize(filenameSMs)
else
  sm = calcSMs(p)
  serialize(filenameSMs, sm)
end

MX = MY = 4

plot2DSM(rfft(reshape(sm[:FluidFNO],:,3,30,30),1), MX, MY; filename="systemMatrixFluidFNO.png")
plot2DSM(rfft(reshape(sm[:Immobilized135FNO],:,3,30,30),1), MX, MY; filename="systemMatrixImmobilized135FNO.png")
plot2DSM(rfft(reshape(sm[:Immobilized45FNO],:,3,30,30),1), MX, MY; filename="systemMatrixImmobilized45FNO.png")
plot2DSM(rfft(reshape(sm[:FluidFokkerPlanck],:,3,30,30),1), MX, MY; filename="systemMatrixFluidFokkerPlanck.png")
plot2DSM(rfft(reshape(sm[:Immobilized135FokkerPlanck],:,3,30,30),1), MX, MY; filename="systemMatrixImmobilized135FokkerPlanck.png")
plot2DSM(rfft(reshape(sm[:Immobilized45FokkerPlanck],:,3,30,30),1), MX, MY; filename="systemMatrixImmobilized45FokkerPlanck.png")

function relError(ŷ::AbstractArray, y::AbstractArray, )
  return mean( NeuralMNP.norm_l2(ŷ.-y, dims=(1,2)) ./ NeuralMNP.norm_l2(y, dims=(1,2)) )
end

@info relError(sm[:FluidFNO], sm[:FluidFokkerPlanck])
@info relError(sm[:Immobilized45FNO], sm[:Immobilized45FokkerPlanck])
@info relError(sm[:Immobilized135FNO], sm[:Immobilized135FokkerPlanck])
