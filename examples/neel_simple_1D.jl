using MNPDynamics
using Plots

# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 11000;         # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
relaxation = NEEL
reltol = 1e-4
abstol = 1e-6

const amplitude = 0.012
const fx = 25000;
tLength = 1000;       # length of time vector
tMax = 4/fx;          # maximum evaluation time in seconds

t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (amplitude*[sin(2*pi*fx*t); 0*t; 0*t]);

@time t, y = simulationMNP(B, t; n, DCore, kAnis, N, reltol, abstol, relaxation)

p1 = plot(t, y[:,1])
plot!(p1, t, y[:,2])
plot!(p1, t, y[:,3])

dt = diff(t);
dxdt = diff(y[:,1])./dt;
dydt = diff(y[:,2])./dt;
dzdt = diff(y[:,3])./dt;
p2 = plot(t[1:end-1], dxdt)
plot!(p2, t[1:end-1], dydt)
plot!(p2, t[1:end-1], dzdt)

plot(p1,p2)