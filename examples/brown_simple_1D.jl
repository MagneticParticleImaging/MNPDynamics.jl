using MNPDynamics
using Plots

# Parameters
DCore = 20e-9;         # particle diameter in nm
DHydro = 20e-9;         # particle diameter in nm
η = 1e-5;
relaxation = BROWN
reltol = 1e-4
abstol = 1e-6

fx = 25000;
tLength = 1000;       # length of time vector
tMax = 4/fx;          # maximum evaluation time in seconds

t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (0.012*[sin(2*pi*fx*t); 0*t; 0*t]);

@time t, y = simulationMNP(B, t; n, DCore, DHydro, η, reltol, abstol, relaxation)

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