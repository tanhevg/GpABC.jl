using Documenter, GpAbc

makedocs(doctest = false)
deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/tanhevg/GpAbc.jl.git",
    julia  = "0.6"
)
