
# Parameters
p = Dict{Symbol,Any}()
#p[:DCore] = (19.999e-9, 20.0001e-9) # particle diameter in nm
p[:DCore] = 20e-9          # particle diameter in nm
p[:α] = 0.1                # damping coefficient
p[:kAnis] = (0,1600)      # anisotropy constant
p[:N] = 20                 # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL      # relaxation mode
p[:reltol] = 1e-4          # relative tolerance
p[:abstol] = 1e-6          # absolute tolerance
p[:tWarmup] = 0.00002      # warmup time
p[:derivative] = false
p[:solver] = :FBDF         # Use more stable solver
p[:trainTimeParam] = false

p[:numData] = 5000
p[:numTrainingData] = round(Int, p[:numData]*0.9)
p[:numTestData] = p[:numData] - p[:numTrainingData]
p[:snippetLength] = 200
p[:samplingRate] = 2.5e6
p[:maxField] = 0.03
p[:filterFactor] = (4,20)

tSnippet = range(0, step=1/p[:samplingRate], length=p[:snippetLength]);
device = cpu