using MNPDynamics
using Plots

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 11000         # anisotropy constant
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:n] = [1;0;0]           # anisotropy axis
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time

# Excitation frequencies
const fx = 2.5e6 / 102
const fy = 2.5e6 / 96

samplingRate = 2.5e6
tLength = lcm(96,102);             # length of time vector
tMax = lcm(96,102) / samplingRate; # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

const amplitude = 0.012

B = (t, offset) -> (amplitude*[sin(2*pi*fx*t); sin(2*pi*fy*t); 0*t] .+ offset )

nOffsets = (5, 5, 1)

offsets = vec([ amplitude.*4.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])

@time smM = simulationMNPMultiParams(B, t, offsets; p...)
