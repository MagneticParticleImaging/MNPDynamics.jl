using MNPDynamics
using Plots

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 25e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 5000*[1;0;0]  # anisotropy constant and anisotropy axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time


amplitude = 0.012
fx = 25000;
tLength = 1000;       # length of time vector
tMax = 1/fx;          # maximum evaluation time in seconds

t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (amplitude*[sin(2*pi*fx*t); 0*t; 0*t]);

@time y = simulationMNP(B, t; p...)

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