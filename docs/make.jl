using Documenter, MNPDynamics

makedocs(
    format = Documenter.HTML(prettyurls = false),
    modules = [MNPDynamics],
    sitename = "MNPDynamics",
    authors = "Tobias Knopp et al.",
    pages = [
      "Home" => "index.md",
      "Overview" => "overview.md",
      "API" => "api.md",
    ],
    warnonly = [:missing_docs]
)

deploydocs(
    repo = "github.com/MagneticParticleImaging/MNPDynamics.jl.git",
    target = "build",
)