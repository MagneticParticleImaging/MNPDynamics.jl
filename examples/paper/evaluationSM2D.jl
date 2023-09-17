using MNPDynamics
using NeuralMNP
using CairoMakie
using FFTW, HDF5
using Serialization
using Flux
using Statistics
using ImageUtils

include("utils.jl")

function calcSMs(p; device=gpu)

  sm = Dict{Symbol,Any}()

  p[:anisotropyAxis] = nothing
  @time sm[:FluidFNO] = calcSM(p; device)
  α = 1.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized135FNO] = calcSM(p; device)
  α = 0.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized45FNO] = calcSM(p; device)

  delete!(p, :neuralNetwork)
  delete!(p, :alg)

  p[:anisotropyAxis] = nothing
  @time sm[:FluidFokkerPlanck] = calcSM(p; device)
  α = 1.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized135FokkerPlanck] = calcSM(p; device)
  α = 0.5
  p[:anisotropyAxis] = (cos(pi/2*α), sin(pi/2*α), 0.0)
  @time sm[:Immobilized45FokkerPlanck] = calcSM(p; device) 

  return sm
end

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
pSM = copy(NOModel.params)
pSM[:DCore] = 20e-9        # particle diameter in nm
pSM[:kAnis] = 1250         # anisotropy constant
pSM[:derivative] = false
pSM[:neuralNetwork] = NOModel
pSM[:alg] = NeuralNetworkMNP
N = 30
pSM[:nOffsets] = (N, N, 1)
pSM[:maxField] = 0.012
pSM[:dividers] = (102,96,1)

filenameSMs = "sm.bin"
if isfile(filenameSMs)
  sm = deserialize(filenameSMs)
else
  sm = calcSMs(pSM, device=gpu)
  serialize(filenameSMs, sm)
end






function plot2DSM(grid, smFT, MX, MY)

  for (iy,my) in enumerate(MY)
    for (ix,mx) in enumerate(MX)
      ax = CairoMakie.Axis(grid[iy, ix], ylabel="", 
          xticklabelsvisible=false, xticksvisible=false)

      A = collect(transpose(ImageUtils.complexColoring(collect(
            transpose(smFT[(mx-1)*16+(my-1)*17+1, 1, :, :, 1])))))

      CairoMakie.heatmap!(ax, A)  
              #c=:viridis, axis=nothing, colorbar=nothing, yflip=false,
              #annotations = (2, 4, Plots.text("$(mx-1),$(my-1)", :white,:left,"Helvetica Bold", 20)) ))
      hidedecorations!(ax, grid=false, label=false)
      tightlimits!(ax)
    end
  end

  rowgap!(grid, 5) 
  colgap!(grid, 5)

  nothing
end

function relErrorMixingOrder(smEstimate, smOrig, maxMixingError )
  err = zeros(maxMixingError,maxMixingError)

  for my = 1:maxMixingError
    for mx = 1:maxMixingError

      A = smEstimate[(mx-1)*16+(my-1)*17+1, 1, :, :, 1]
      B = smOrig[(mx-1)*16+(my-1)*17+1, 1, :, :, 1]

      err[my, mx] = norm(A-B) / maximum(abs.(B)) / sqrt(length(B))
      #err[my, mx] = abs(assess_ssim(A, B))
    end
  end
  return err
end

function plotAll()

  MX = 1:2:9
  MY = 1:2:9
  maxMixingError = 9
  strDataset = ["fluid", "immobilized 45°", "immobilized 135°"]

  fig = Figure( resolution = (1000, 950), figure_padding = 1 )#, fontsize = 16)

  gr1 = GridLayout()
  fig[1,1] = gr1
  gr2 = GridLayout()
  fig[1,2] = gr2
  gr3 = GridLayout()
  fig[1,3] = gr3
  gr4 = GridLayout()
  fig[2,1] = gr4
  gr5 = GridLayout()
  fig[2,2] = gr5
  gr6 = GridLayout()
  fig[2,3] = gr6

  smOrig = [rfft(reshape(sm[:FluidFokkerPlanck],:,3,N,N),1),
             rfft(reshape(sm[:Immobilized135FokkerPlanck],:,3,N,N),1),
             rfft(reshape(sm[:Immobilized45FokkerPlanck],:,3,N,N),1)]

  smEstim = [rfft(reshape(sm[:FluidFNO],:,3,N,N),1),
             rfft(reshape(sm[:Immobilized135FNO],:,3,N,N),1),
             rfft(reshape(sm[:Immobilized45FNO],:,3,N,N),1)]

  plot2DSM(gr1, smOrig[1], MX, MY)
  plot2DSM(gr2, smOrig[2], MX, MY )
  plot2DSM(gr3, smOrig[3], MX, MY)

  plot2DSM(gr4, smEstim[1], MX, MY)
  plot2DSM(gr5, smEstim[2], MX, MY )
  plot2DSM(gr6, smEstim[3], MX, MY)



  CairoMakie.Axis(fig[1,1], ylabel = L"m_y", yreversed = true, 
     ylabelpadding = 1, xticks = MX, yticks = MY, limits=(0.1,9.9,0.1,9.9),
     xticksvisible = false, xticklabelsvisible = false)
  CairoMakie.Axis(fig[2,1], xlabel=L"m_x", ylabel = L"m_y", yreversed = true, 
     ylabelpadding = 1, xticks = MX, yticks = MY, limits=(0.1,9.9,0.1,9.9))
  CairoMakie.Axis(fig[2,2], xlabel=L"m_x", yreversed = true, 
     ylabelpadding = 1, xticks = MX, yticks = MY, limits=(0.1,9.9,0.1,9.9),
     yticksvisible = false, yticklabelsvisible = false)
  CairoMakie.Axis(fig[2,3], xlabel=L"m_x", yreversed = true, 
     ylabelpadding = 1, xticks = MX, yticks = MY, limits=(0.1,9.9,0.1,9.9),
     yticksvisible = false, yticklabelsvisible = false)   
     
  for l=1:length(strDataset)
    Label(fig[1, l, Top()], strDataset[l], valign = :bottom, font = :bold, #fontsize = 24,
         padding = (0, 0, 5, 0))
  end
  Label(fig[1, 1, Left()], "Fokker-Planck", halign = :left, font = :bold, #fontsize = 24,
         padding = (0, 0, 0, 0), rotation=π/2)
  Label(fig[2, 1, Left()], "FNO", halign = :left, font = :bold, #fontsize = 24,
         padding = (0, 0, 0, 0), rotation=π/2)
  
  rowgap!(fig.layout, 20) 
  colgap!(fig.layout, 20)
       

  C = collect(range(0,1,length=100)) *
  transpose(exp.(2*π*im*collect(range(0,1,length=100))))
  axleg, pleg = CairoMakie.heatmap(fig[1:2,4],
      (complexColoring(C)),
       axis=(ylabel="phase", xlabel="amplitude", #title="colorbar", titlefont = :bold,
      #titlesize = 24, 
      yaxisposition = :right,
      xticks=([1,100],[L"0",L"1"]),
      yticks=([0.5,50,100.5],[L"0", L"\pi", L"2\pi"])))
  colsize!(fig.layout, 4, Aspect(1, 0.1))

  # Error Plots
  crange = (0.0,0.2)
  for l=1:length(smOrig)

    err = relErrorMixingOrder(smEstim[l], smOrig[l], maxMixingError )
    ax = CairoMakie.Axis(fig[3, l], xlabel=L"m_x",
           ylabel = (l==1) ? L"m_y" : "",
           title = "$(strDataset[l])",
           yreversed = true,
           yticklabelsvisible=(l==1), yticksvisible=(l==1))

    @info maximum(err)
    CairoMakie.heatmap!(ax, err, colorrange = crange   )  
    #hidedecorations!(ax, grid=false, label=false)
    tightlimits!(ax)
  end

  fig[3,4] = Colorbar(fig[3, 3], limits = crange, label = "NRMSD")

  #Label(fig[4, 1, Left()], "relative error", halign = :left, font = :bold, #fontsize = 24,
  #       padding = (0, 0, 0, 0), rotation=π/2)

  save("img/systemmatrixComparison2D.pdf", fig, pt_per_unit = 0.25)
  
end
plotAll()

#plot2DSM(rfft(reshape(sm[:FluidFokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixFluidFokkerPlanck.png")
#plot2DSM(rfft(reshape(sm[:Immobilized135FokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized135FokkerPlanck.png")
#plot2DSM(rfft(reshape(sm[:Immobilized45FokkerPlanck],:,3,N,N),1), MX, MY; filename="systemMatrixImmobilized45FokkerPlanck.png")

#@info relError(sm[:FluidFNO], sm[:FluidFokkerPlanck])
#@info relError(sm[:Immobilized45FNO], sm[:Immobilized45FokkerPlanck])
#@info relError(sm[:Immobilized135FNO], sm[:Immobilized135FokkerPlanck])


