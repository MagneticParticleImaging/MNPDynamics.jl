
# Parameters
DCore = 20e-9;         # particle diameter in nm
DHydro = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 11000;         # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
relaxation = BROWN
reltol = 1e-4
abstol = 1e-6

fx = 25000;
tLength = 1000;       # length of time vector
tMax = 4/fx;          # maximum evaluation time in seconds

t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (0.012*[sin(2*pi*fx*t); 0*t; 0*t]);

t, y = simulationMNP(B, t; n, DCore, DHydro, kAnis, N, reltol, abstol, relaxation)
