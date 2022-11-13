


function smTest(params, anisotropyAxis=nothing)

  tLengthSM = lcm(96,102);             # length of time vector
  tMaxSM = lcm(96,102) / samplingRate; # maximum evaluation time in seconds
  
  tSM = range(0, stop=tMaxSM, length=tLengthSM);
  
  fx = 2.5e6 / 102
  fy = 2.5e6 / 96
  BSM = (t, offset) -> (maxField*[sin(2*pi*fx*t), sin(2*pi*fy*t), 0] .+ offset )
  
  nOffsets = (30, 30, 1)
  
  oversampling = 1.25
  offsets = vec([ oversampling*maxField.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  if anisotropyAxis == nothing
    anisotropyAxis = vec([ oversampling*2.0.*((collect(Tuple(x)).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  else
    anisotropyAxis = vec([ anisotropyAxis  for x in CartesianIndices(nOffsets) ])
  end
  params[:kAnis] = 1250*anisotropyAxis

  sm = simulationMNPMultiParams(BSM, tSM, offsets; params...)

  return sm
end

pSM = Dict{Symbol,Any}()
pSM[:DCore] = 20e-9        # particle diameter in nm
pSM[:kAnis] = 1250         # anisotropy constant
pSM[:derivative] = false
pSM[:neuralNetwork] = NOModel2

@time sm = smTest(pSM);
smPred = reshape(sm,:,3,30,30);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 8, 8; filename="systemMatrixPredict2.svg")


α = 1.5
@time sm = smTest(pSM, [1.1*cos(pi/2*α), 1.1*sin(pi/2*α), 0.0]);



plot2DSM(smMFTPred, 8, 8; filename="systemMatrixPredict.svg")