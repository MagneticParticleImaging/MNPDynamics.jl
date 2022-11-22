using MNPDynamics
using BSON
using Plots, StatsPlots
using Flux, NeuralOperators
using LinearAlgebra

include("params.jl")

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = (15e-9, 25e-9) # particle diameter in nm
p[:α] = 0.1                # damping coefficient
p[:kAnis] = (0,1250)       # anisotropy constant
p[:N] = 20                 # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL      # relaxation mode
p[:reltol] = 1e-4          # relative tolerance
p[:abstol] = 1e-6          # absolute tolerance
p[:tWarmup] = 0.00005      # warmup time
p[:derivative] = false
p[:solver] = :FBDF         # Use more stable solver

Z = 5000
ZTest = 100
snippetLength = 200
samplingRate = 2.5e6
tMax = snippetLength / samplingRate; # maximum evaluation time in seconds

tSnippet = range(0, stop=tMax, length=snippetLength);

maxField = 0.012

filenameTrain = "trainData.h5"
filenameModel = "model.bin"

BTrain, pTrain = generateStructuredFields(p, tSnippet, Z; maxField=2*maxField, fieldType=RANDOM_FIELD)
mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tSnippet, pTrain)

X, Y = prepareTrainData(pTrain, tSnippet, BTrain, mTrain; useTime = true)

NOModel = deserialize(filenameModel)


function plotExampleSignals(BTrain, mTrain, t)
  
  numSignals = 4
  p1 = plot(t.*1000, BTrain[:,1,end], lw=2, c=1, title="Magnetic Fields", legend=nothing,
            xlabel="t / ms", ylabel="a.u")
  for z = 2:numSignals
    plot!(p1, t.*1000,  BTrain[:,1,end-z+1], lw=2, c=z)
  end

  p2 = plot(t.*1000, mTrain[:,1,end], lw=2, c=1, title="Magnetic Moments", legend=nothing,
            xlabel="t / ms", ylabel="a.u")
  for z = 2:numSignals
    plot!(p2, t.*1000, mTrain[:,1,end-z+1], lw=2, c=z)
  end

  p = plot(p1, p2, layout=(2,1), size=(600,400))
  savefig(p, "exampleSignals.pdf")
end
plotExampleSignals(BTrain, mTrain, tSnippet)


function plotErrorStatistics(p, neuralNetwork, X, Y, t)
  ND = 6
  NK = 5
  intervalD = range(p[:DCore][1], p[:DCore][2], length=ND)
  intervalK = range(p[:kAnis][1], p[:kAnis][2], length=NK)
  
  numSignals = 2000

  error = zeros(ND, NK)
  errorCount = ones(ND, NK)

  for z = 1:numSignals
    kAnis = norm(X[1,5:7,z])
    D = X[1,4,z]

    xc = X[:,:,z:z]
    xc = trafo(xc, neuralNetwork.normalizationX)
    yc = back(neuralNetwork.model(Float32.(xc)), neuralNetwork.normalizationY)
    err = norm(yc - Y[:,:,z:z]) / norm(Y[:,:,z:z])

    d = round(Int, (D-p[:DCore][1]) / (p[:DCore][2]-p[:DCore][1]) * (ND-1) + 1)
    k = round(Int, (kAnis-p[:kAnis][1]) / (p[:kAnis][2]-p[:kAnis][1]) * (NK-1) + 1)

    #@info typeof(err)
    error[d,k] += err
    errorCount[d,k] += 1
  end

  error = error ./ errorCount

  p_ = heatmap(error', c=:viridis, xlabel="DCore", ylabel="kAnis", xticks=(1:ND,round.(collect(intervalD).*1e9, digits=2)),
  yticks=(1:NK,round.(collect(intervalK), digits=0)))

  savefig(p_, "statisticalError.pdf")
end


plotErrorStatistics(p, NOModel, X, Y, tSnippet)

