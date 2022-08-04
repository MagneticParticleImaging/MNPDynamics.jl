using MNPDynamics
using Plots

# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 11000;         # anisotropy constant
t_length = 1000;       # length of time vector
t_max = 2/25000;       # maximum evaluation time in seconds
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
reltol = 1e-3
abstol = 1e-6

t = range(0,stop=t_max,length=t_length);


f = 25000;
# Magnetic field for simulation 
B =  t -> (0.012*[sin(2*pi*f*t); 0*t; 0*t]);

@time t, y = simulationMNP(B, t; n, DCore, kAnis, N, reltol, abstol)

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