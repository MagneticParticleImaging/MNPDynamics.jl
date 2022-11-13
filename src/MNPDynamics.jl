module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using Distributed
using SharedArrays

using ProgressMeter

using HDF5, BSON
using Flux, NeuralOperators, MLUtils
using Flux: withgradient
using StatsBase #, Statistics
using Random
using ImageFiltering
using Interpolations
using Plots
using FFTW

#import Pardiso
#using Sundials

@enum RelaxationType begin
  NEEL
  BROWN
  NO_RELAXATION
end

export NEEL, BROWN, NO_RELAXATION

@enum FieldType begin
  RANDOM_FIELD
  HARMONIC_RANDOM_FIELD
  HARMONIC_MPI_FIELD
end

export NEEL, RANDOM_FIELD, HARMONIC_RANDOM_FIELD, HARMONIC_MPI_FIELD

include("utils.jl")
include("sparseMatrixSetup.jl")
include("simulation.jl")
include("neuralOperator.jl")
include("visualization.jl")


end # module
