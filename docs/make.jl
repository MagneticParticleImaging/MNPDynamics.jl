using Documenter, MNPDynamics

makedocs(
    modules = [MNPDynamics],
    format = :html,
    checkdocs = :exports,
    sitename = "MNPDynamics.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/tknopp/MNPDynamics.jl.git",
)
