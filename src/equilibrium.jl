
function simulationMNP(B::g, tVec, ::EquilibriumModel;
                       MS = 474000.0, 
                       DCore = 20e-9, 
                       temp = 293.0,
                       derivative = false,
                       kargs...
                       ) where g
    

  y = zeros(Float64, length(tVec), 3)

  dt = step(tVec)/10
                  
  if !derivative
    for ti=1:length(tVec)
      y[ti, :] = langevin(B(tVec[ti]); DCore, temp, MS)
    end
  else
    for ti=1:length(tVec)
      y[ti, :] = (langevin(B(tVec[ti]+dt); DCore, temp, MS)-langevin(B(tVec[ti]); DCore, temp, MS)) / dt
    end
  end
                  
  return y
end
