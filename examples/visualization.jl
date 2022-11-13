using Plots

function plot2DSM(smFT, MX, MY; filename=nothing)
  pl = Any[]
  for my=1:MY
    for mx=1:MX
      push!(pl, heatmap(abs.(smFT[(mx-1)*16+(my-1)*17+1, 1, :, :, 1]), 
              c=:viridis, axis=nothing, colorbar=nothing,
              annotations = (2, 4, Plots.text("$mx,$my", :white,:left)) ))
    end
  end
  plot(pl..., layout=(MX,MY), size=(1000,1000))
  if filename != nothing
    savefig(filename)
  end
end