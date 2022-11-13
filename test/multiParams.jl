@testset "Multi Params" begin


# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 1100*[1;0;0]   # anisotropy constant and anisotropy axis
N = 20;                # maximum spherical harmonics index to be considered
relaxation = NEEL
reltol = 1e-3
abstol = 1e-6

# Excitation frequencies
fb = 2.5e6
fx = fb / 102
fy = fb / 96

samplingMultiplier = 2                          # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*lcm(96,102)        # length of time vector
tMax = (lcm(96,102)-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0, stop=tMax, length=tLength);

# Magnetic field for simulation
amplitude = 0.012
B = (t, offset) -> SVector{3,Float64}(amplitude*cospi(2*fx*t)+offset[1], amplitude*cospi(2*fy*t)+offset[2], offset[3])

nOffsets = (5, 1, 1)

offsets = vec([ amplitude.*4.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])

smM = simulationMNPMultiParams(B, t, offsets; DCore, kAnis, N, reltol, abstol, relaxation)


end