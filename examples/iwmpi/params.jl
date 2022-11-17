

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = (19.999e-9, 20.0001e-9) # particle diameter in nm
p[:α] = 0.1                # damping coefficient
p[:kAnis] = (0,1500)       # anisotropy constant
p[:N] = 20                 # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL      # relaxation mode
p[:reltol] = 1e-4          # relative tolerance
p[:abstol] = 1e-6          # absolute tolerance
p[:tWarmup] = 0.00005      # warmup time
p[:derivative] = false
p[:solver] = :FBDF         # Use more stable solver

Z = 5000
ZTrain = round(Int, Z*0.8)
ZTest = Z - ZTrain
snippetLength = 200
samplingRate = 2.5e6
tMax = snippetLength / samplingRate; # maximum evaluation time in seconds

tSnippet = range(0, stop=tMax, length=snippetLength);
maxField = 30e-3
