using Plots, ImageUtils

function plot2DSM(smFT, MX, MY; filename=nothing)
  pl = Any[]
  for my=1:MY
    for mx=1:MX
      push!(pl, heatmap(ImageUtils.complexColoring(collect(
          transpose(smFT[(mx-1)*16+(my-1)*17+1, 1, :, :, 1]))), 
              c=:viridis, axis=nothing, colorbar=nothing, yflip=false,
              annotations = (2, 4, Plots.text("$(mx-1),$(my-1)", :white,:left,"Helvetica Bold", 20)) ))
    end
  end
  plot(pl..., layout=(MX,MY), size=(1000,1000))
  if filename != nothing
    savefig(filename)
  end
end