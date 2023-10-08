
"""
  simulationMNP(B, t, offsets; kargs...) 
"""
function simulationMNP(B::g, t; kargs...) where g

  # This might need to be more clever
  alg = get(kargs, :model, FokkerPlanckModel())
  return simulationMNP(B, t, alg; kargs...)
end

"""
    simulationMNP(B::Matrix{T}, t; kargs...) 

This version takes the fields in discretized form
"""
function simulationMNP(B::Matrix{T}, t; kargs...) where {T}

  function fieldInterpolator(field, t)
    function help_(field, t, d)
      itp = interpolate(field[:,d], BSpline(Cubic(Flat(OnCell()))))
      etp = extrapolate(itp, Flat())
      sitp = scale(etp, t)
      return sitp
    end
    return (help_(field,t,1), help_(field,t,2), help_(field,t,3))
  end

  magneticMoments = zeros(Float64, length(t), 3)

  let param = fieldInterpolator(B, t)
    BFunc = (t) -> ( [param[1](t), param[2](t), param[3](t)] )
    magneticMoments[:,:] .= simulationMNP(BFunc, t; kargs...)
  end
  return magneticMoments
end