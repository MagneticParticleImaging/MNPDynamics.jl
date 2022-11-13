@testset "Accuracy" begin

eps = 1e-2

# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 0;             # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
DHydro = 20e-9;         # particle diameter in nm
η = 1e-5;

reltol = 1e-4
abstol = 1e-6

fx = 25000;
tLength = 1000;       # length of time vector
tMax = 4/fx;          # maximum evaluation time in seconds

t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (0.012*[sin(2*pi*fx*t); 0*t; 0*t]);

yLangevin = simulationMNP(B, t; DCore, relaxation = NO_RELAXATION)
yNeel = simulationMNP(B, t; DCore, kAnis, N, relaxation = NEEL,
                                  reltol, abstol)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

yBrown = simulationMNP(B, t; DCore, DHydro, η, relaxation = BROWN,
                           reltol, abstol)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

end
