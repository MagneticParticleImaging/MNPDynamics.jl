module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using ImageFiltering
using Interpolations
using OffsetArrays

using Distributed
using SharedArrays
using StaticArrays

using ProgressMeter

using HDF5

using FLoops
using FFTW

using SpecialFunctions

## sorting algorithms ##

abstract type AbstractMNPModel end

struct FokkerPlanckModel <: AbstractMNPModel end
struct EquilibriumModel <: AbstractMNPModel end
struct EquilibriumAnisModel <: AbstractMNPModel end

export AbstractMNPModel, FokkerPlanckModel, EquilibriumModel, EquilibriumAnisModel

@enum RelaxationType begin
  NEEL
  BROWN
  NO_RELAXATION
end

export NEEL, BROWN, NO_RELAXATION

export EnsembleThreads, EnsembleDistributed, EnsembleSerial

include("utils.jl")
include("sparseMatrixSetup.jl")
include("fokkerPlanck.jl")
include("equilibrium.jl")
include("equilibriumAnis.jl")
include("simulation.jl")
include("multiParams.jl")

export simulationMNP, simulationMNPMultiParams, loadSimulationMNPMultiParams


end # module
