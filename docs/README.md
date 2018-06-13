# Documentation

Documentation is generated from the docstrings in the source.
See [Julia Manual](https://docs.julialang.org/en/stable/manual/documentation/) on docstring syntax.

Building documentation is a two step process.

First use [Documenter](https://juliadocs.github.io/Documenter.jl/stable/) to pull
out all the doc strings from the source into a single Markdown file, and generate the html. From `docs`, run the command

    $ julia make.jl local

This will generate the local web site. The results can be checked by opening `build/index.html` in the web browser. If you want to test one file only,
and don't want to wait for the entire source tree to be processed, add

    Pages=["your_file.jl"]

under the `Modules ... ` line in `src/reference.md`.

To generate a LaTeX pdf, use [Pandoc](https://pandoc.org/). LaTeX must be installed on your machine.

    $ pandoc build/reference.md -o gauss_pro_abc_doc.tex

#### LaTeX Tips:
- [LaTeX syntax documentation](https://juliadocs.github.io/Documenter.jl/stable/man/latex/#Julia-0.5-1)
- Backslashes in LaTeX commands in docstrings must be escaped properly (double backslash, `\\`).
- Use LaTeX commands for Greek letters, not unicode symbols (`\\theta`, not `θ`)
