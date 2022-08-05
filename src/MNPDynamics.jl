module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using Distributed
using SharedArrays

using ProgressMeter

#import Pardiso
#using Sundials

@enum RelaxationType begin
  NEEL
  BROWN
end

export NEEL, BROWN

include("utils.jl")
include("sparseMatrixSetup.jl")
include("simulation.jl")


end # module
