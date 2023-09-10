using MNPDynamics, NeuralMNP
using Serialization
using CairoMakie
using Flux, NeuralOperators
using DataFrames

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
include("params.jl")
include("utils.jl")

function plotExampleSignals(model)
  
  kAnis = 4000
  offset = [0.0,0,0]
  DCore = 24*1e-9


  p = Dict{Symbol,Any}()
  p[:α] = 0.1               # damping coefficient
  p[:kAnis] = kAnis*[1.0,0.0,0.0] # anisotropy constant and anisotropy axis
  p[:N] = 20                # maximum spherical harmonics index to be considered
  p[:relaxation] = NEEL     # relaxation mode
  p[:reltol] = 1e-4         # relative tolerance
  p[:abstol] = 1e-6         # absolute tolerance
  p[:tWarmup] = 0.00005     # warmup time
  p[:derivative] = true   
  
  amplitude = 0.015
  fx = 12500;
  tMax = 1/fx; 
  samplingRate = model.params[:samplingRate] * 10

  tLength = round(Int, tMax*samplingRate);  

  t = range(0,step=1/samplingRate,length=tLength);
  lw = 1.5
  

  μ₀ = 4*π*1e-7
  fig = Figure(  resolution = (600, 1000), figure_padding = 1 )#, fontsize = 16)
  ax1 = CairoMakie.Axis(fig[1, 1], title = "Diameter", ylabel=L"\frac{\textrm{d}m}{\textrm{d}t} / (\frac{\textrm{MAm}^{2}}{s})", xticklabelsvisible=false, xticksvisible=false)
  ax2 = CairoMakie.Axis(fig[2, 1], title = "Anisotropy", ylabel=L"\frac{\textrm{d}m}{\textrm{d}t} / (\frac{\textrm{MAm}^{2}}{s})", xticklabelsvisible=false, xticksvisible=false)
  ax3 = CairoMakie.Axis(fig[3, 1], title = "Field Offset", ylabel=L"\frac{\textrm{d}m}{\textrm{d}t} / (\frac{\textrm{MAm}^{2}}{s})", xticklabelsvisible=false, xticksvisible=false)
  ax4 = CairoMakie.Axis(fig[4, 1], title = "Field Amplitude", ylabel=L"\frac{\textrm{d}m}{\textrm{d}t} / (\frac{\textrm{MAm}^{2}}{s})", xticklabelsvisible=false, xticksvisible=false)
  ax5 = CairoMakie.Axis(fig[5, 1], title = "Field Frequency", xlabel=L"t / \textrm{ms}", ylabel=L"\frac{\textrm{d}m}{\textrm{d}t} / (\frac{\textrm{MAm}^{2}}{s})")

  linkxaxes!(ax1, ax2, ax3, ax4, ax5)
  CairoMakie.xlims!(ax5, 0, maximum(t))

  
  DCores = collect(16.5:2.5:24)
  for d=1:length(DCores)
    p[:DCore] = DCores[d]*1e-9         # particle diameter in nm

    # Magnetic field for simulation 
    B = t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ offset);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)

    pNO[:neuralNetwork] = model
    pNO[:alg] = NeuralNetworkMNP
    yNO = simulationMNP(B, t; pNO...)

    CairoMakie.lines!(ax1, t[:], y[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[d]...), linewidth=lw, 
                 linestyle = :dot)
    CairoMakie.lines!(ax1, t[:], yNO[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[d]...), linewidth=lw, label=L"D=%$(DCores[d])\,\text{nm}",
                 linestyle = :solid)

  end


  kAnis_ = collect([0, 1500, 3000, 4000])

  p[:DCore] = DCore
  for k=1:length(kAnis_)
    p[:kAnis] = kAnis_[k]*[1.0;0.0;0] #NeuralMNP.randAxis()  # anisotropy constant and anisotropy axis

    # Magnetic field for simulation 
    B =  t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ offset);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:alg] = NeuralNetworkMNP
    yNO = simulationMNP(B, t; pNO...)

    CairoMakie.lines!(ax2, t[:], y[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, 
                 linestyle = :dot)
    CairoMakie.lines!(ax2, t[:], yNO[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, label=L"K_\text{anis}=%$(kAnis_[k])\,\text{Jm}^{-3}",
                 linestyle = :solid)
  end


  p[:DCore] = DCore
  p[:kAnis] = kAnis*[1;0;0]
  off = collect([-15, -7.5, 7.5, 16])

  for k=1:length(off)

    # Magnetic field for simulation 
    B =  t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ [off[k]*1e-3, 0.0, 0.0]);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:alg] = NeuralNetworkMNP
    yNO = simulationMNP(B, t; pNO...)

    CairoMakie.lines!(ax3, t[:], y[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, 
        linestyle = :dot)

    CairoMakie.lines!(ax3, t[:], yNO[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, label=L"H_\text{offset}=%$(off[k])\,\text{mT}",
                 linestyle = :solid)
  end

  p[:DCore] = DCore
  p[:kAnis] = kAnis*[1;0;0]
  ampl = collect([0.005, 0.01, 0.015, 0.02])

  for k=1:length(ampl)

    # Magnetic field for simulation 
    B =  t -> (ampl[k]*[-cos(2*pi*fx*t); 0*t; 0*t] )

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:alg] = NeuralNetworkMNP
    yNO = simulationMNP(B, t; pNO...)

    CairoMakie.lines!(ax4, t[:], y[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, 
                          linestyle = :dot)
    CairoMakie.lines!(ax4, t[:], yNO[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, label=L"H_\text{amplitude}=%$(ampl[k]*1000)\,\textrm{mT}",
                 linestyle = :solid)
  end

  p[:DCore] = DCore
  p[:kAnis] = kAnis*[1;0;0]
  freq = collect([fx/2, fx, fx*2, fx*3])

  for k=1:length(freq)

    # Magnetic field for simulation 
    B =  t -> (amplitude*[-cos(2*pi*freq[k]*t); 0*t; 0*t] )

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:alg] = NeuralNetworkMNP
    yNO = simulationMNP(B, t; pNO...)

    CairoMakie.lines!(ax5, t[:], y[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, 
                 linestyle = :dot)
    CairoMakie.lines!(ax5, t[:], yNO[:,1]/1e6, 
        color = CairoMakie.RGBf(colors[k]...), linewidth=lw, label=L"f=%$(freq[k]/1000)\,\textrm{kHz}",
                 linestyle = :solid)
  end

  CairoMakie.ylims!(ax5, -1.7, 2.7)

  axislegend(ax1; position = :rt, nbanks = 2, labelsize=13, padding=(3,3,3,3), rowgap = 1)
  axislegend(ax2; position = :rt, nbanks = 2, labelsize=13, padding=(3,3,3,3), rowgap = 1)
  axislegend(ax3; position = :rt,  nbanks = 2, labelsize=13, padding=(3,3,3,3), rowgap =1)
  axislegend(ax4; position = :rt,  nbanks = 2, labelsize=13, padding=(3,3,3,3), rowgap =1)
  axislegend(ax5; position = :rt,  nbanks = 4, labelsize=13, padding=(3,3,3,3), rowgap =1)

  save("img/evalMPS.pdf", fig)
  fig
end
plotExampleSignals(NOModel)