using MNPDynamics
using Plots, Plots.Measures, StaticArrays


c = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)]

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9                  # particle diameter in nm
p[:relaxation] = NO_RELAXATION     # relaxation mode

amplitude = 0.012
fx = 25000;
tLength = 1000;       # length of time vector
tMax = 1/fx;          # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

# Magnetic field for simulation 
B(t) = amplitude*[cos(2*pi*fx*t), 0, 0]

m = simulationMNP(B, t; p...)

p1 = plot(t, 1000*getindex.(B.(t),1), legend=nothing, title="Magnetic Field", lw=2, c=c[1], ylabel="B / mT")
p2 = plot(t, m[:,1], legend=nothing, title="Magnetic Moment", lw=2, c=c[2], ylabel="a.u.")
dxdt = diff(m[:,1])./diff(t)/1e5;
p3 = plot(t[1:end-1], dxdt, title="Derivative Magnetic Moment", legend=nothing, lw=2, c=c[3],  
                    ylabel="a.u.", xlabel="time / s", bottom_margin=5mm)
plot(p1, p2, p3, layout=(3,1), right_margin=5mm, size=(800,400))
savefig("simpleNoRelaxation.svg")



##########################

p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 11000*[1;0;0] # anisotropy constant and axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time

m = simulationMNP(B, t; p...)

p1 = plot(t, 1000*getindex.(B.(t),1), legend=nothing, title="Magnetic Field", lw=2, c=c[1], ylabel="B / mT")
p2 = plot(t, m[:,1], legend=nothing, title="Magnetic Moment", lw=2, c=c[2], ylabel="a.u.")
dxdt = diff(m[:,1])./diff(t)/1e5;
p3 = plot(t[1:end-1], dxdt, title="Derivative Magnetic Moment", legend=nothing, lw=2, c=c[3],  
                    ylabel="a.u.", xlabel="time / s", bottom_margin=5mm)
plot(p1, p2, p3, layout=(3,1), right_margin=5mm, size=(800,400))
savefig("neelRelaxation.svg")



##########################

p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 11000*[1;0;0] # anisotropy constant and axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00001     # warmup time

# Magnetic field for simulation 
Brect(t) = amplitude*[ 0.25 < fx*mod(t,1) < 0.75  ? -1.0 : 1.0 , 0, 0]

m = simulationMNP(Brect, t; p...)

p1 = plot(t, 1000*getindex.(Brect.(t),1), legend=nothing, title="Magnetic Field", lw=2, c=c[1], ylabel="B / mT")
p2 = plot(t, m[:,1], legend=nothing, title="Magnetic Moment", lw=2, c=c[2], ylabel="a.u.")
dxdt = diff(m[:,1])./diff(t)/1e5;
p3 = plot(t[1:end-1], dxdt, title="Derivative Magnetic Moment", legend=nothing, lw=2, c=c[3],  
                    ylabel="a.u.", xlabel="time / s", bottom_margin=5mm)
plot(p1, p2, p3, layout=(3,1), right_margin=5mm, size=(800,400))
savefig("neelRelaxationRect.svg")



######################


# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:DHydro] = 80e-9        # particle diameter in nm
p[:η] = 1e-5              # viscosity
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = BROWN    # relaxation mode
p[:reltol] = 1e-4         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time

m = simulationMNP(B, t; p...)

p1 = plot(t, 1000*getindex.(B.(t),1), legend=nothing, title="Magnetic Field", lw=2, c=c[1], ylabel="B / mT")
p2 = plot(t, m[:,1], legend=nothing, title="Magnetic Moment", lw=2, c=c[2], ylabel="a.u.")
dxdt = diff(m[:,1])./diff(t)/1e5;
p3 = plot(t[1:end-1], dxdt, title="Derivative Magnetic Moment", legend=nothing, lw=2, c=c[3],  
                    ylabel="a.u.", xlabel="time / s", bottom_margin=5mm)
plot(p1, p2, p3, layout=(3,1), right_margin=5mm, size=(800,400))
savefig("brownRelaxation.svg")


######################


# Excitation frequencies
fb = 2.5e6
fx = fb / 102

samplingMultiplier = 2                  # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*102        # length of time vector
tMax = (102-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0,stop=tMax,length=tLength);

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 18e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 10000*[1;0;0]  # anisotropy constant and anisotropy axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = BROWN     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 1 / fb      # warmup time

amps = [0.002, 0.005, 0.01, 0.02, 0.04]

# No Relaxation

BNoR =  t -> SVector{3,Float64}(amps[end]*[cospi(2*fx*t), 0, 0])
pNoRelax = copy(p)
pNoRelax[:relaxation] = NO_RELAXATION
m = simulationMNP(BNoR, t; pNoRelax...)
BNoRelax = getindex.(BNoR.(t),1)
mNoRelax = m[:,1]

# With Brown Relaxation 

msBrown = []
BsBrown = []

for l=1:length(amps)
  B_ =  t -> SVector{3,Float64}(amps[l]*[cospi(2*fx*t), 0, 0])
  m_ = simulationMNP(B_, t; p...)

  push!(BsBrown, getindex.(B_.(t),1)) 
  push!(msBrown, m_[:,1])

end

# With Neel Relaxation 

msNeel = []
BsNeel = []

p[:relaxation] = NEEL

for l=1:length(amps)
  B_ =  t -> SVector{3,Float64}(amps[l]*[cospi(2*fx*t), 0, 0])
  m_ = simulationMNP(B_, t; p...)

  push!(BsNeel, getindex.(B_.(t),1)) 
  push!(msNeel, m_[:,1])

end

c = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05), RGB(0.5,0.5,0.5)]

p1 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops Brown Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / a.u.", xlabel="H / T", bottom_margin=5mm, label="No Relax")

p2 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops Neel Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / a.u", xlabel="H / T", bottom_margin=5mm, label="No Relax")

for l=1:length(amps)
  plot!(p1, BsBrown[l],msBrown[l], lw=2, c=c[l], label = "Amp = $(amps[l])")
  plot!(p2, BsNeel[l],msNeel[l], lw=2, c=c[l], label = "Amp = $(amps[l])")
end

plot(p1, p2, layout=(2,1), right_margin=5mm, left_margin=5mm, size=(1000,600))
savefig("hysteresis.svg")
