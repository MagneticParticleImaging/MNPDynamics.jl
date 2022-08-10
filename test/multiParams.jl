@testset "Multi Params" begin


# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 11000;         # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
relaxation = NEEL
reltol = 1e-3
abstol = 1e-6

# Excitation frequencies
fx = 2.5e6 / 102
fy = 2.5e6 / 96

samplingRate = 2.5e6
tLength = lcm(96,102);             # length of time vector
tMax = lcm(96,102) / samplingRate; # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

amplitude = 0.012

B = (t, offset) -> (amplitude*[sin(2*pi*fx*t); sin(2*pi*fy*t); 0*t] .+ offset )

nOffsets = (5, 1, 1)

offsets = vec([ amplitude.*4.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])

smM = simulationMNPMultiParams(B, t, offsets; DCore, kAnis, N, reltol, abstol, relaxation)


end