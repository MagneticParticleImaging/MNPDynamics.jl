using MNPDynamics
using Plots, StaticArrays

# Excitation frequencies
fb = 2.5e6
fx = fb / 102
fy = fb / 96
fz = fb / 99

samplingMultiplier = 2                             # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*lcm(96,99,102)        # length of time vector
tMax = (lcm(96,99,102)-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0, stop=tMax, length=tLength);

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 1100*[1;0;0]  # anisotropy constant and anisotropy axis
p[:N] = 10                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-5         # relative tolerance
p[:abstol] = 1e-5         # absolute tolerance
p[:tWarmup] = 1 / fb      # warmup time

# Magnetic field for simulation
const amplitude = 0.012
B = t -> SVector{3,Float64}(amplitude*[cospi(2*fx*t), cospi(2*fy*t), cospi(2*fz*t)])

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