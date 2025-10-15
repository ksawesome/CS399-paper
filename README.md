# CS399-paper: Socratic LLM Benchmarking for Mettle

This repository contains the full source, data, and build instructions for the CS399 paper: benchmarking Large Language Models (LLMs) for Socratic tutoring in the Mettle platform.

## Project Overview
- **Goal:** Select and rigorously benchmark an LLM to serve as an adaptive Socratic chatbot for engineering estimation education.
- **Framework:** Multi-dimensional evaluation of candidate models on pedagogical quality, contextual adaptability, cost, reliability, and integration effort.
- **Models Evaluated:**
  - OpenAI gpt-4o-mini
  - Anthropic claude-3-sonnet-20240229
  - Google gemini-2.5-flash
  - Cohere command-r-08-2024
  - Meta Llama-4-Maverick-17B-128E-Instruct
- **Methodology:** Automated Python benchmarking harness, curated prompt bank, mixed-methods scoring (automated metrics + human rubrics).

## Repository Structure
```
LICENSE
main.tex
preamble.sty
README.md
references.bib
appendices/
  ...
build/
  ...
code/
data/
figures/
sections/
  01_Abstract.tex
  02_Introduction.tex
  ...
```
- `main.tex`: Main LaTeX file, imports all sections.
- `preamble.sty`: Custom style and package configuration.
- `sections/`: Paper sections (abstract, methods, results, etc.).
- `appendices/`: Supplementary material (rubrics, prompt bank, schemas, etc.).
- `code/`: Benchmarking scripts and harness (see paper for details).
- `data/`: Experimental data and results.
- `figures/`: Plots and diagrams for the paper.
- `build/`: LaTeX build output (PDF, aux files).

## Build Instructions
1. **Requirements:**
   - LaTeX distribution (MiKTeX or TeX Live 2024+)
   - Python 3.8+
   - [Pygments](https://pygments.org/) (for minted code listings)
2. **Compile the paper:**
   - Run: `latexmk -pdf -outdir=build main.tex`
   - If using minted, ensure `-shell-escape` is enabled and `pygmentize` is installed.
3. **View output:**
   - Final PDF: `build/main.pdf`

## Notes
- All code listings use the `minted` package for syntax highlighting. If you encounter errors, ensure `pygmentize` is installed and no deprecated minted options are present in `preamble.sty`.
- For a comparison of `minted` vs `listings`, see the paper or ask for a summary.

## License
This work is licensed under the MIT License. See `LICENSE` for details.

## Contact
For questions or collaboration, open an issue or contact the repository owner.

