using MNPDynamics
using Plots, Plots.Measures, Random
using FFTW

p = Dict{Symbol,Any}()
p[:DCore] = 25e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 5000*[1;0;0]  # anisotropy constant and axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-4         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00001     # warmup time

amplitude = 0.012
fx = 25000;
tLength = 200;        # length of time vector
tMax = 1/fx;          # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

Random.seed!(0)
# Magnetic field for simulation 
BRand = randn(tLength, 3)
BRandFT = rfft(BRand)
BRandFT[50:end,:] .= 0
BRand = irfft(BRandFT, tLength)
BRand = amplitude * BRand ./ maximum(abs.(BRand))


@time m = simulationMNP(BRand, t; p...)

c = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)]
p1 = plot(t, 1000*BRand[:,1], legend=nothing, title="Magnetic Field", lw=2, c=c[1], ylabel="B / mT")
p2 = plot(t, m[:,1], legend=nothing, title="Magnetic Moment", lw=2, c=c[2], ylabel="a.u.")
dxdt = diff(m[:,1])./diff(t)/1e5;
p3 = plot(t[1:end-1], dxdt, title="Derivative Magnetic Moment", legend=nothing, lw=2, c=c[3],  
                    ylabel="a.u.", xlabel="time / s", bottom_margin=5mm)
plot(p1, p2, p3, layout=(3,1), right_margin=5mm, size=(800,400))
