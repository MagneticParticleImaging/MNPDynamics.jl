module NeuralMNP
  using LinearAlgebra
  using MNPDynamics

  using HDF5, BSON
  using Flux, NeuralOperators, MLUtils
  using Flux: withgradient
  using StatsBase #, Statistics
  using Random



  include("neuralOperator.jl")
end