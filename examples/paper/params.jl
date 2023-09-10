
# Parameters
p = Dict{Symbol,Any}()
#p[:DCore] = (19.999e-9, 20.001e-9) # particle diameter in nm
p[:DCore] = (15.000e-9, 25.000e-9) # particle diameter in nm
p[:α] = 0.1                # damping coefficient
#p[:kAnis] = (1000,1500)       # anisotropy constant
p[:kAnis] = (0,5000)       # anisotropy constant
p[:N] = 20                 # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL      # relaxation mode
p[:reltol] = 1e-4          # relative tolerance
p[:abstol] = 1e-6          # absolute tolerance
p[:tWarmup] = 0.00002      # warmup time
p[:derivative] = false
p[:solver] = :FBDF         # Use more stable solver
p[:trainTimeParam] = false
p[:samplingRate] = 2.5e6

p[:numBaseData] = 50000
p[:baseDataLength] = 200
p[:snippetLength] = 200
splits = (0.85, 0.05, 0.1)
p[:numBaseValidationData] = round(Int, p[:numBaseData]*splits[2])
p[:numBaseTestData] = round(Int, p[:numBaseData]*splits[3])
p[:numBaseTrainingData] = p[:numBaseData] - p[:numBaseValidationData] - p[:numBaseTestData]

p[:numTrainingData] = floor(Int,p[:numBaseTrainingData] * p[:baseDataLength] / p[:snippetLength])  # 400
p[:numValidationData] = floor(Int,p[:numBaseValidationData] * p[:baseDataLength] / p[:snippetLength]) 
p[:numTestData] = floor(Int,p[:numBaseTestData] * p[:baseDataLength] / p[:snippetLength]) 


tBaseData = range(0, step=1/p[:samplingRate], length=p[:baseDataLength]);
tSnippet = range(0, step=1/p[:samplingRate], length=p[:snippetLength]);
device = gpu
datadir = "./data"
imgdir = "./img"
modeldir = "./models"

mkpath(datadir)
mkpath(imgdir)

forceDataGen = false
seed = 2

#=numDatasets = 4
dfDatasets = DataFrame(samplingDistribution = [:chi, :chi, :uniform, :uniform],
                       fieldDims = [1:3, 1, 1:3, 1],
                       anisotropyAxis = [nothing, [1,0,0], nothing, [1,0,0]],
                       fieldType = repeat([RANDOM_FIELD], numDatasets),
                       filterFactor = repeat([(17,24)], numDatasets),
                       maxField = repeat([0.03], numDatasets),
                       numData = repeat([p[:numBaseData]], numDatasets),
                       filename = ["trainData$(i).h5" for i=1:numDatasets])=#

numDatasets = 2
dfDatasets = DataFrame(samplingDistribution = [:chi, :chi],
                      fieldDims = [1:3, 1],
                      anisotropyAxis = [nothing, [1,0,0]],
                      fieldType = repeat([RANDOM_FIELD], numDatasets),
                      filterFactor = repeat([(10,20)], numDatasets),
                      maxField = repeat([0.03], numDatasets),
                      numData = repeat([p[:numBaseData]], numDatasets),
                      filename = ["trainData$(i).h5" for i=1:numDatasets])





dfTraining = DataFrame()


defaultNetworkParams = (
  networkModes = 18,
  networkWidth = 48,
)

defaultTrainingParams = (
  η = 1f-3,
  γ = 0.5f0,
  stepSize = 30,
  epochs = 300,
  batchSize = 20
)

defaultWeighting = (0.9, 0.1)

defaultNumTrainingData = p[:numTrainingData]

### Study 1: Compare distributions for fixed

# Weighting Study 
weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for d=1:length(weights)
  push!(dfTraining, (;
       datasetTraining = (1,2),
       datasetTrainingWeighting = (1.0-weights[d], weights[d]),
       numTrainingData = defaultNumTrainingData,
       datasetValidation = (1, 2),
       defaultNetworkParams...,
       defaultTrainingParams...
      ) 
    )
end

# Num Data Study 
numData = [1000, 5000, 10000, 20000, 50000]

for d=1:length(numData)
  push!(dfTraining, (;
       datasetTraining = (1,2),
       datasetTrainingWeighting = defaultWeighting,
       numTrainingData = numData[d],
       datasetValidation = (1, 2),
       defaultNetworkParams...,
       defaultTrainingParams...
      ) 
    )
end



# Sampling Study
push!(dfTraining, (;
      datasetTraining = (1,2),
      datasetTrainingWeighting = defaultWeighting,
      numTrainingData = defaultNumTrainingData,
      datasetValidation = (1, 2),
      defaultNetworkParams...,
      defaultTrainingParams...
    ) 
  )

push!(dfTraining, (;
    datasetTraining = (1,2),
    datasetTrainingWeighting = defaultWeighting,
    numTrainingData = defaultNumTrainingData,
    datasetValidation = (1, 2),
    defaultNetworkParams...,
    defaultTrainingParams...
  ) 
)


# Architecture Study 
networkModes = [8, 16, 24, 32]
networkWidth = [16, 32, 48, 64]


for d1=1:length(networkModes)
  for d2=1:length(networkWidth)
    push!(dfTraining, (;
       datasetTraining = (1,2),
       datasetTrainingWeighting = (1.0-weights[d], weights[d]),
       numTrainingData = defaultNumTrainingData,
       datasetValidation = (1, 2),
       networkModes = networkModes[d1],
       networkWidth = networkWidth[d2],
       defaultTrainingParams...
      ) 
    )
  end
end



