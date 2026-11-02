#!/usr/bin/env bash
# Build paper/cpc/manuscript.pdf: pdflatex -> bibtex -> pdflatex -> pdflatex.
#
# MiKTeX installed per-user does not create the plain `pdflatex` launcher on
# every setup, so fall back to driving the engine directly with --undump.
set -e
cd "$(dirname "$0")"

BIN="${MIKTEX_BIN:-$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64}"
if [ -x "$BIN/pdflatex.exe" ]; then
    LATEX=("$BIN/pdflatex.exe")
else
    LATEX=("$BIN/miktex-pdftex.exe" --undump=pdflatex)
fi
BIBTEX="$BIN/miktex-bibtex.exe"
[ -x "$BIN/bibtex.exe" ] && BIBTEX="$BIN/bibtex.exe"

run() { echo "  $*"; "$@" -interaction=nonstopmode -halt-on-error manuscript.tex > /dev/null; }

echo "pass 1 (latex)";  run "${LATEX[@]}"
echo "pass 2 (bibtex)"; "$BIBTEX" manuscript > /dev/null
echo "pass 3 (latex)";  run "${LATEX[@]}"
echo "pass 4 (latex)";  run "${LATEX[@]}"

echo
grep -c "Overfull" manuscript.log 2>/dev/null | sed 's/^/overfull boxes: /' || true
tail -n 2 manuscript.log | head -n 1
