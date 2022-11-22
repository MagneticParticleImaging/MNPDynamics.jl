using MNPDynamics
using NeuralMNP
using Plots, Measures
using FFTW, HDF5
using ProgressMeter

include("params.jl")
include("../visualization.jl")


function smTest(params, maxField, samplingRate, anisotropyAxis=nothing)

  tLengthSM = lcm(96,99,102);             # length of time vector
  tMaxSM = lcm(96,99,102) / samplingRate; # maximum evaluation time in seconds
  
  tSM = range(0, step=1/samplingRate, length=tLengthSM);
  
  fx = 2.5e6 / 102
  fy = 2.5e6 / 96
  fz = 2.5e6 / 99
  BSM = (t, offset) -> (maxField*[sin(2*pi*fx*t), sin(2*pi*fy*t), sin(2*pi*fz*t)] .+ offset )
  
  nOffsets = (30, 30, 30)
    
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

    @showprogress for z=1:prod(nOffsets)

      off = offsets[z]
      Bx = [BSM(t_, off)[1] for t_ in tSM]
      By = [BSM(t_, off)[2] for t_ in tSM]
      Bz = [BSM(t_, off)[3] for t_ in tSM]
      n_ = anisotropyAxis[z]

      #t_ = range(0,1,length=tLengthSM)

      XLongTest = cat( Bx, By, Bz, n_[1]*ones(Float32, tLengthSM), n_[2]*ones(Float32, tLengthSM),
                      n_[3]*ones(Float32, tLengthSM), dims=2)

      q = NeuralMNP.applyToArbitrarySignal(params[:neuralNetwork], XLongTest)
      sm[:,:,z] .= q
    end
    return sm
  else
    sm = simulationMNPMultiParams(BSM, tSM, offsets; params...)
    return sm  
  end
  
end


filenameModel = "model.bin"
NOModel = deserialize(filenameModel)

p[:neuralNetwork] = NOModel
@time sm = smTest(p, maxField, samplingRate);
