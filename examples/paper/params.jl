
# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = (15.000e-9, 25.000e-9)  # particle diameter in nm
p[:α] = 0.1                         # damping coefficient
p[:kAnis] = (0,5000)                # anisotropy constant
p[:N] = 20                          # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL               # relaxation mode
p[:reltol] = 1e-4                   # relative tolerance
p[:abstol] = 1e-6                   # absolute tolerance
p[:tWarmup] = 0.00002               # warmup time
p[:derivative] = false
p[:solver] = :FBDF                  # Use more stable solver
p[:trainTimeParam] = false

p[:numBaseData] = 500
p[:baseDataLength] = 400
p[:snippetLength] = 200
splits = (0.9, 0.1, 0.1)
p[:numBaseValidationData] = round(Int, p[:numBaseData]*splits[2])
p[:numBaseTestData] = round(Int, p[:numBaseData]*splits[3])
p[:numBaseTrainingData] = p[:numBaseData] - p[:numBaseValidationData] - p[:numBaseTestData]

p[:numTrainingData] = floor(Int, p[:numBaseTrainingData] * p[:baseDataLength] / p[:snippetLength])  # 400
p[:numValidationData] = floor(Int,p[:numBaseValidationData] * p[:baseDataLength] / p[:snippetLength]) 



p[:samplingRate] = 2.5e6

tBaseData = range(0, step=1/p[:samplingRate], length=p[:baseDataLength]);
tSnippet = range(0, step=1/p[:samplingRate], length=p[:snippetLength]);
device = gpu
datadir = "./data"
imgdir = "./img"

mkpath(datadir)
mkpath(imgdir)

numDatasets = 4
dfDatasets = DataFrame(samplingDistribution = [:chi, :chi, :uniform, :uniform],
                       fieldDims = [1:3, 1, 1:3, 1],
                       anisotropyAxis = [nothing, [1,0,0], nothing, [1,0,0]],
                       fieldType = repeat([RANDOM_FIELD], numDatasets),
                       filterFactor = repeat([(17,24)], numDatasets),
                       maxField = repeat([0.03], numDatasets),
                       numData = repeat([p[:numBaseData]], numDatasets),
                       filename = ["trainData$(i).h5" for i=1:numDatasets])

#=numDatasets = 2
dfDatasets = DataFrame(samplingDistribution = [:chi, :chi],
                      fieldDims = [1:3, 1],
                      anisotropyAxis = [nothing, [1,0,0]],
                      fieldType = repeat([RANDOM_FIELD], numDatasets),
                      filterFactor = repeat([(17,24)], numDatasets),
                      maxField = repeat([0.03], numDatasets),
                      numData = repeat([p[:numBaseData]], numDatasets),
                      filename = ["trainData$(i).h5" for i=1:numDatasets])=#
