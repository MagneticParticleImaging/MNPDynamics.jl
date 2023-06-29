

function calcSM(params; device=cpu)

  dividers = pSM[:dividers]
  tLengthSM = lcm(dividers...);             # length of time vector
  tMaxSM = tLengthSM / params[:samplingRate]; # maximum evaluation time in seconds
  
  tSM = range(0, step=1/params[:samplingRate], length=tLengthSM);
  
  freq = ntuple( d -> dividers[d] > 1 ? params[:samplingRate] ./ dividers[d] : 0.0, 3)
  ampl = ntuple( d -> dividers[d] > 1 ? params[:maxField] : 0.0, 3)

  BSM = (t, offset) -> MNPDynamics.SVector{3,Float32}(ampl[1]*sin(2*pi*freq[1]*t)+offset[1], 
                                          ampl[2]*sin(2*pi*freq[2]*t)+offset[2], 
                                          ampl[3]*sin(2*pi*freq[3]*t)+offset[3] )
  nOffsets = pSM[:nOffsets]
  factor = pSM[:kAnis]
  anisotropyAxis = get(pSM, :anisotropyAxis, nothing)
  
  oversampling = 1.25
  offsets = vec([ oversampling*params[:maxField].*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  if anisotropyAxis == nothing
    anisotropyAxis = vec([ factor.*oversampling*2.0.*(((Tuple(x)).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  else
    anisotropyAxis = vec([ Tuple(factor.*anisotropyAxis)  for x in CartesianIndices(nOffsets) ])
  end
  params_ = copy(params)
  params_[:kAnis] = anisotropyAxis

  sm = simulationMNPMultiParams(BSM, tSM, offsets; device, params_...)
  return sm 
end

function relError(ŷ::AbstractArray, y::AbstractArray)
  return mean( NeuralMNP.norm_l2(ŷ.-y, dims=(1,2)) ./ NeuralMNP.norm_l2(y, dims=(1,2)) )
end