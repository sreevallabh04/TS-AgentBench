#!/usr/bin/env bash
#
# Build the manuscript from source: LaTeX, BibTeX, LaTeX, LaTeX.
#
#   bash scripts/build_paper.sh
#
# Two passes follow BibTeX because the first resolves citation keys and the second
# settles page numbers and cross-references that moved as a result. Skipping the last
# pass leaves "??" in the PDF without failing the build, which is the kind of error that
# is only ever noticed by a reviewer.
#
# The build is deliberately strict: any LaTeX error, undefined reference or undefined
# citation fails the script. A manuscript whose numbers are generated
# (scripts/make_paper_assets.py) but whose build is only eyeballed would be checked at
# the wrong end.
#
# Regenerate tables and macros first if results have changed:
#   python scripts/make_paper_assets.py && python scripts/verify_manuscript.py

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPERDIR="$HERE/paper"
DOC="${1:-main}"

# MiKTeX installs per user and is not always on PATH in a non-login shell. Candidates
# are normalised to POSIX form first: under MSYS a mixed path like C:\Users\...\bin
# satisfies `test -x` but is not usable as a PATH entry, so an unnormalised candidate
# silently produces a PATH that looks right and resolves nothing.
posix_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1" 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}

for candidate in \
  "$HOME/AppData/Local/Programs/MiKTeX/miktex/bin/x64" \
  "${LOCALAPPDATA:-}/Programs/MiKTeX/miktex/bin/x64" \
  "/c/Program Files/MiKTeX/miktex/bin/x64" \
  "/usr/local/texlive/bin" ; do
  [ -n "$candidate" ] || continue
  candidate="$(posix_path "$candidate")"
  if [ -x "$candidate/pdflatex.exe" ] || [ -x "$candidate/pdflatex" ]; then
    PATH="$candidate:$PATH"
    break
  fi
done
export PATH

command -v pdflatex >/dev/null 2>&1 || {
  echo "pdflatex not found. Install a TeX distribution (MiKTeX or TeX Live)," >&2
  echo "or build paper/ on Overleaf. On Windows: winget install MiKTeX.MiKTeX" >&2
  exit 127
}

cd "$PAPERDIR" || exit 1
LOGDIR="$PAPERDIR/.build"
mkdir -p "$LOGDIR"

run() {
  local label="$1"; shift
  if ! "$@" > "$LOGDIR/$label.log" 2>&1; then
    # BibTeX warns about entries it cannot sort; that is not fatal and is reported
    # rather than treated as failure.
    if [ "$label" = "bibtex" ]; then
      echo "note: bibtex reported warnings (see $LOGDIR/bibtex.log)"
      return 0
    fi
    echo "FAILED at $label; first errors:" >&2
    grep -E "^!" "$LOGDIR/$label.log" | head -20 >&2
    exit 1
  fi
}

run pass1  pdflatex -interaction=nonstopmode "$DOC.tex"
run bibtex bibtex "$DOC"
run pass2  pdflatex -interaction=nonstopmode "$DOC.tex"
run pass3  pdflatex -interaction=nonstopmode "$DOC.tex"

FINAL="$LOGDIR/pass3.log"
undefined_refs=$(grep -c "Reference .* undefined" "$FINAL" || true)
undefined_cites=$(grep -c "Citation .* undefined" "$FINAL" || true)
overfull=$(grep -c "Overfull .hbox" "$FINAL" || true)
pages=$(grep -oE "Output written on $DOC\.pdf \([0-9]+ pages" "$FINAL" \
        | grep -oE "[0-9]+" | head -1)

echo "built $PAPERDIR/$DOC.pdf (${pages:-?} pages)"
echo "  undefined references : $undefined_refs"
echo "  undefined citations  : $undefined_cites"
echo "  overfull hboxes      : $overfull"

if [ "$undefined_refs" -gt 0 ] || [ "$undefined_cites" -gt 0 ]; then
  echo "refusing to pass with unresolved references or citations" >&2
  exit 1
fi
echo "OK"
