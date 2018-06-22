using Documenter, GpABC

local_build = "local" in ARGS

makedocs(
    format=:html,
    sitename = "GpABC.jl",
    authors = "Tara Abdul Hameed, Fei He, Jonathan Ish-Horowitz, Istvan Klein, Elizabeth Roesch, Evgeny Tankhilevich",
    pages = [
        "Home" => "index.md",
        "examples.md",
        "Reference" => "reference.md"
    ],
    html_prettyurls = !local_build
    # html_canonical = "https://tanhevg.github.io/GpABC.jl/stable/",  TODO
)

if !local_build
    deploydocs(
        # deps   = Deps.pip("mkdocs", "python-markdown-math"),
        deps   = nothing,
        make   = nothing,
        target = "build",
        repo   = "github.com/tanhevg/GpABC.jl.git",
        julia  = "0.6.2"
    )
end
