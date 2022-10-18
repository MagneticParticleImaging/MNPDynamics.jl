module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using ImageFiltering
using OffsetArrays

using Distributed
using SharedArrays

using ProgressMeter

#import Pardiso
#using Sundials

@enum RelaxationType begin
  NEEL
  BROWN
  NO_RELAXATION
end

export NEEL, BROWN, NO_RELAXATION

include("utils.jl")
include("sparseMatrixSetup.jl")
include("simulation.jl")


end # module
