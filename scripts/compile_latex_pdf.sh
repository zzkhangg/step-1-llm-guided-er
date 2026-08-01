#!/usr/bin/env bash
set -euo pipefail

tex_file="${1:-reports/weekly_sampler_progress_2026-06-16.tex}"

if [[ "$tex_file" != /* ]]; then
    tex_file="$PWD/$tex_file"
fi

if [[ ! -f "$tex_file" ]]; then
    echo "LaTeX file not found: $tex_file" >&2
    exit 1
fi

if [[ "${tex_file##*.}" != "tex" ]]; then
    echo "Expected a .tex file: $tex_file" >&2
    exit 1
fi

tex_dir="$(cd "$(dirname "$tex_file")" && pwd -P)"
tex_name="$(basename "$tex_file")"
pdf_name="${tex_name%.tex}.pdf"

if command -v latexmk >/dev/null 2>&1; then
    compiler="latexmk"
elif command -v pdflatex >/dev/null 2>&1; then
    compiler="pdflatex"
else
    cat >&2 <<'EOF'
No LaTeX compiler found.

Install one of:
  - latexmk, recommended
  - pdflatex

Examples:
  Ubuntu/Debian: sudo apt-get install latexmk texlive-latex-base texlive-latex-recommended
  macOS:         brew install --cask mactex-no-gui
EOF
    exit 127
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/latex-build.XXXXXX")"

cleanup() {
    status=$?
    if [[ $status -eq 0 ]]; then
        rm -rf "$build_dir"
    else
        echo "Build files kept for debugging: $build_dir" >&2
    fi
}
trap cleanup EXIT

cd "$tex_dir"

if [[ "$compiler" == "latexmk" ]]; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir="$build_dir" "$tex_name"
else
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$build_dir" "$tex_name"
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$build_dir" "$tex_name"
fi

if [[ ! -f "$build_dir/$pdf_name" ]]; then
    echo "Expected PDF was not produced: $build_dir/$pdf_name" >&2
    exit 1
fi

cp "$build_dir/$pdf_name" "$tex_dir/$pdf_name"
echo "Wrote $tex_dir/$pdf_name"
