@testset "Accuracy" begin

eps = 1e-2

# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 0;             # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
DHydro = 20e-9;         # particle diameter in nm
η = 1e-5;

reltol = 1e-4
abstol = 1e-6

fb = 2.5e6
fx = fb / 102

samplingMultiplier = 2                          # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*102                # length of time vector
tMax = (102-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> SVector{3,Float64}(0.012*[sin(2*pi*fx*t), 0*t, 0*t]);

yLangevin = simulationMNP(B, t; n, DCore, relaxation = NO_RELAXATION)
yNeel = simulationMNP(B, t; n, DCore, kAnis, N, relaxation = NEEL,
                                  reltol, abstol)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

yBrown = simulationMNP(B, t; n, DCore, DHydro, η, relaxation = BROWN,
                           reltol, abstol)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

end
