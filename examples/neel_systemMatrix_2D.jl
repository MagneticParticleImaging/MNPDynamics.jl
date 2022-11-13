using MNPDynamics
using Plots, StaticArrays, FFTW

# Excitation frequencies
fb = 2.5e6
fx = fb / 102
fy = fb / 96

samplingMultiplier = 2                          # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*lcm(96,102)        # length of time vector
tMax = (lcm(96,102)-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0, stop=tMax, length=tLength);

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 625           # anisotropy constant
p[:N] = 10                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-5         # relative tolerance
p[:abstol] = 1e-5         # absolute tolerance
p[:tWarmup] = 1 / fb      # warmup time
p[:derivative] = true

# Magnetic field for simulation
const amplitude = 0.012
B = (t, offset) -> SVector{3,Float64}(amplitude*cospi(2*fx*t)+offset[1], amplitude*cospi(2*fy*t)+offset[2], offset[3])

nOffsets = (20, 20, 1)

oversampling = 1.25
offsets = vec([ oversampling*amplitude.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
anisotropyAxis = vec([ oversampling*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])

p[:kAnis] = p[:kAnis]*anisotropyAxis

@time smM = simulationMNPMultiParams(B, t, offsets; p...)

smMFT = reshape(rfft(smM, 1), :, 3, nOffsets...)

plot2DSM(smMFT, 8, 8; filename="systemMatrix.svg")
