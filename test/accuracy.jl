@testset "Accuracy" begin

eps = 1e-2

p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:DHydro] = 20e-9        # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 0*[1;0;0]     # anisotropy constant and axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time
p[:η] = 1e-5;

fb = 2.5e6
fx = fb / 102

samplingMultiplier = 2                          # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*102                # length of time vector
tMax = (102-1/samplingMultiplier) / fb          # maximum evaluation time in seconds
t = range(0,stop=tMax,length=tLength);

# Magnetic field for simulation 
B =  t -> (0.012*[sin(2*pi*fx*t), 0*t, 0*t]);

p[:model] = EquilibriumModel()
yLangevin = simulationMNP(B, t; p...)

p[:model] = FokkerPlanckModel()
p[:relaxation] = NEEL
yNeel = simulationMNP(B, t; p...)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

p[:relaxation] = BROWN
yBrown = simulationMNP(B, t; p...)

e = norm(yLangevin[:] - yNeel[:]) / norm(yLangevin[:])
@test e < eps 

end
