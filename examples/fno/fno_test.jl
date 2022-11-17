using MNPDynamics
using NeuralMNP
using Plots, Measures
using FFTW, HDF5
using Flux, NeuralOperators, MLUtils
using LinearAlgebra
using ProgressMeter
using Flux: withgradient
using StatsBase, Statistics
using Random
using Images
using Interpolations

include("../visualization.jl")

const tLength = 200
Z = 5000
ZTrain = round(Int, Z*0.9)
ZTest = Z - ZTrain

filenameData = "training_data_anisotropy_axis.h5"


# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
kAnis = 1250              # anisotropy constant
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-4         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time
p[:derivative] = false
p[:solver] = :FBDF        # Use more stable solver

const samplingRate = 2.5e6
const tMax = tLength / samplingRate; # maximum evaluation time in seconds

const fx = 2.5e6 / 102
const fy = 2.5e6 / 96
const fz = 2.5e6 / 99
const t = range(0, stop=tMax, length=tLength);
const tLong = range(-tMax, stop=tMax, length=2*tLength);
const tVeryLong = range(0, stop=10*tMax, length=10*tLength);

const amplitude = 0.012
fields = rand(range(-1,1,length=1000),2*tLength,3,Z)
for z=1:Z
  for l=1:3
    filtFactor = rand(range(4,20,length=40))
    fields[:,l,z] = imfilter(fields[:,l,z],Kernel.gaussian((filtFactor,))) 
    fields[:,l,z] ./= maximum(abs.(fields[:,l,z]))
    fields[:,l,z] .= amplitude*(rand()*fields[:,l,z] .+ 
                                0.5*rand(range(-1,1,length=1000))*ones(Float32,2*tLength))
  end
end

B = (t, param) -> ( [param[1](t), param[2](t), param[3](t)] )

function genInterpolator(field, t)
  itpx = interpolate(field[:,1], BSpline(Cubic(Periodic(OnCell()))))
  etpx = extrapolate(itpx, Periodic())
  sitpx = scale(etpx, t)
  itpy = interpolate(field[:,2], BSpline(Cubic(Periodic(OnCell()))))
  etpy = extrapolate(itpy, Periodic())
  sitpy = scale(etpy, t)
  itpz = interpolate(field[:,3], BSpline(Cubic(Periodic(OnCell()))))
  etpz = extrapolate(itpz, Periodic())
  sitpz = scale(etpz, t)
  return (sitpx,sitpy,sitpz)
end

function randAxis()
  n = rand(3)*2 .-1
  return n / norm(n) * rand() # in sphere
end

params = vec([ genInterpolator(fields[:,:,z], tLong)  for z=1:Z ])
anisotropyAxis = vec([ randAxis() for z=1:Z ]) 
p[:kAnis] =  kAnis*anisotropyAxis

BLAS.set_num_threads(1)
@time smM = simulationMNPMultiParams(B, t, params; p...)


######

inputChan = 7
outputChan = 3

X = zeros(Float32, tLength, inputChan, Z)
Y = zeros(Float32, tLength, outputChan, Z)

for z=1:Z
  p_ = params[z]
  n_ = anisotropyAxis[z]
  Bx = [B(t_, p_)[1] for t_ in t]
  By = [B(t_, p_)[2] for t_ in t]
  Bz = [B(t_, p_)[3] for t_ in t]

  t_ = range(0,1,length=tLength)

  X[:,:,z] = cat( Bx, By, Bz, n_[1]*ones(Float32, tLength), n_[2]*ones(Float32, tLength),
                  n_[3]*ones(Float32, tLength), t_, dims=2)
  Y[:,:,z] = smM[:,:,z]
end

nX = normalizeData(X; dims=(1,3))
nY = normalizeData(Y; dims=(1,3))

X .= NeuralMNP.trafo(X, nX)
Y .= NeuralMNP.trafo(Y, nY)


bs = 20# 4

#trainData, testData = splitobs((X,Y), at=(0.9, ), shuffle=true)
#trainLoader = DataLoader(trainData, batchsize=bs, shuffle=true)
#testLoader = DataLoader(testData, batchsize=bs, shuffle=true)

trainLoader = DataLoader((X[:,:,1:ZTrain],Y[:,:,1:ZTrain]), batchsize=bs, shuffle=true)
testLoader = DataLoader((X[:,:,(ZTrain+1):end],Y[:,:,(ZTrain+1):end]), batchsize=bs, shuffle=false)

modes = 12 #24
width = 32

model = NeuralMNP.make_neural_operator_model(inputChan, outputChan, modes, width, NeuralMNP.NeuralOperators.FourierTransform)

#opt = Flux.ADAM(1f-3)
η = 1f-3
γ = 0.5
stepSize = 30
epochs = 100

opt = Flux.Optimiser(ExpDecay(η, γ, stepSize, 1f-5), Adam())

#train(model, opt, trainLoader, testLoader, Ymean, Ystd; epochs)
NeuralMNP.train(model, opt, trainLoader, testLoader, nY; epochs)

NOModel = NeuralMNP.NeuralNetwork(model, nX, nY, Dict{Symbol,Any}(), tLength)


function smTest(NOModel, anisotropyAxis=nothing)

  tLengthSM = lcm(96,102);             # length of time vector
  tMaxSM = lcm(96,102) / samplingRate; # maximum evaluation time in seconds
  
  tSM = range(0, step=1/samplingRate, length=tLengthSM);
  
  BSM = (t, offset) -> (amplitude*[sin(2*pi*fx*t), sin(2*pi*fy*t), 0] .+ offset )
  
  nOffsets = (30, 30, 1)
  
  oversampling = 1.25
  offsets = vec([ oversampling*amplitude.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  if anisotropyAxis == nothing
    anisotropyAxis = vec([ oversampling*2.0.*(((Tuple(x)).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
  else
    anisotropyAxis = vec([ Tuple(anisotropyAxis)  for x in CartesianIndices(nOffsets) ])
  end

  sm = zeros(Float32, tLengthSM,  3, prod(nOffsets))

  for z=1:prod(nOffsets)

    off = offsets[z]
    Bx = [BSM(t_, off)[1] for t_ in tSM]
    By = [BSM(t_, off)[2] for t_ in tSM]
    Bz = [BSM(t_, off)[3] for t_ in tSM]
    n_ = anisotropyAxis[z]

    t_ = range(0,1,length=tLengthSM)

    XLongTest = cat( Bx, By, Bz, n_[1]*ones(Float32, tLengthSM), n_[2]*ones(Float32, tLengthSM),
                    n_[3]*ones(Float32, tLengthSM), t_, dims=2)

    #q = applyArbitrarySignal(model, XLongTest, tLength, Xmean, Xstd, Ymean, Ystd)
    q = NeuralMNP.applyToArbitrarySignal(NOModel, XLongTest)
    sm[:,:,z] .= q
  end
  return sm
end


@time sm = smTest(NOModel);
smPred = reshape(sm,:,3,30,30);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 5, 5; filename="systemMatrixFluid_.png")

α = 1.5
@time sm = smTest(NOModel, (1.1*cos(pi/2*α), 1.1*sin(pi/2*α), 0.0));
smPred = reshape(sm,:,3,30,30);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 5, 5; filename="systemMatrixImmobilized135_.png")

α = 0.5
@time sm = smTest(NOModel, (1.1*cos(pi/2*α), 1.1*sin(pi/2*α), 0.0));
smPred = reshape(sm,:,3,30,30);
smMFTPred = rfft(smPred,1);
plot2DSM(smMFTPred, 5, 5; filename="systemMatrixImmobilized45_.png")