#!/usr/bin/env python
"""Static checks on the manuscript that a LaTeX compile would catch, without LaTeX.

    python scripts/verify_manuscript.py            # report
    python scripts/verify_manuscript.py --strict   # non-zero exit on any problem

The manuscript's integrity rule is that no experimental number is typed by hand: every
inline figure is a macro written by ``scripts/make_paper_assets.py``. That rule has a
failure mode --- a macro used in the prose but never generated --- which LaTeX reports
as an undefined control sequence, and which is easy to introduce when a results script
is renamed or a policy disappears from a run. This script catches it, along with the
other structural faults that only show up at build time:

* macros used in the prose but not defined in ``generated/macros.tex``
* macros whose names contain digits (illegal in a LaTeX control sequence)
* ``\\ref``/``\\label`` mismatches
* ``\\cite`` keys with no entry in ``references.bib``
* ``\\input`` targets that do not exist
* leftover ``\\todo`` markers, reported separately since author-only placeholders are
  expected until submission
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import PAPER_DIR, TABLES_DIR  # noqa: E402

#: Control sequences provided by LaTeX itself or by the loaded packages. A macro used in
#: the prose is only a problem if it is neither built in nor generated, so this list
#: exists to keep the report free of false positives rather than to be exhaustive.
KNOWN = {
    # TeX conditionals. The manuscript defines \ifcameraready so that
    # back-matter text naming the author stays out of an anonymous submission;
    # \newif creates the matching true/false forms, which no static scan
    # can see being defined.
    "newif", "ifcameraready", "camerareadytrue", "camerareadyfalse", "else", "fi",
    # graphicx and TeX length primitives, used by ittable to scale a generated
    # table down to the text block when a new model family widens it.
    "resizebox", "ifdim", "width", "linewidth", "textwidth", "fittable",
    # structure
    "documentclass", "usepackage", "begin", "end", "section", "subsection",
    "subsubsection", "paragraph", "label", "ref", "cite", "citep", "citet", "input",
    "include", "bibliography", "bibliographystyle", "maketitle", "title", "author",
    "date", "newcommand", "renewcommand", "providecommand", "def", "item", "caption",
    "centering", "includegraphics", "graphicspath", "appendix", "footnote",
    # text
    "emph", "textbf", "textit", "texttt", "textsc", "textwidth", "small", "footnotesize",
    "large", "Large", "LARGE", "noindent", "par", "medskip", "bigskip", "smallskip",
    "vspace", "hspace", "newline", "clearpage", "newpage", "quad", "qquad", "times",
    "dagger", "ddagger", "star", "%", "&", "_", "$", "#", "{", "}",
    # math
    "frac", "sum", "prod", "int", "sqrt", "alpha", "beta", "gamma", "delta", "epsilon",
    "tau", "sigma", "mu", "lambda", "theta", "pi", "rho", "phi", "psi", "omega",
    "Delta", "Sigma", "Omega", "Lambda", "approx", "leq", "geq", "neq", "cdot", "ldots",
    "dots", "text", "mathrm", "mathbb", "mathcal", "log", "exp", "min", "max",
    # tables and floats
    "toprule", "midrule", "bottomrule", "hline", "multicolumn", "multirow", "tabular",
    "table", "figure", "hfill", "url", "href", "todo", "textcolor", "colorbox",
    "openreview", "month", "year", "thanks", "and", "AND", "hsize", "linewidth",
    "citeauthor", "citeyear", "citealp", "citealt", "newblock", "bibitem",
    # conditionals and engine primitives used in the preamble
    "IfFileExists", "typeout", "PackageWarning", "CurrentOption", "DeclareOption",
    # math sizing and accents
    "big", "Big", "bigg", "left", "right", "hat", "bar", "tilde", "vec", "ell", "in",
    "mid", "forall", "exists", "subset", "cup", "cap", "setminus", "to", "mapsto",
    "langle", "rangle", "lvert", "rvert", "lVert", "rVert", "top", "bot",
}

#: ``\IfFileExists{path}{then}{else}`` guards content that renders only when a generated
#: artifact is present -- the manuscript uses it so that a section describing results
#: cannot appear when the results do not exist. The checker has to honour it, or it
#: reports the unrendered branch as broken.
IFFILEEXISTS_RE = re.compile(r"\\IfFileExists\{([^}]*)\}")

MACRO_RE = re.compile(r"\\([A-Za-z@]+)")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:page)?ref\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
INPUT_RE = re.compile(r"\\input\{([^}]*)\}")
#: Deliberately permissive about digits so that an *illegal* generated name is captured
#: and reported, rather than slipping past the scan because it does not look like a
#: macro definition.
NEWCMD_RE = re.compile(r"\\newcommand\{\\([A-Za-z@][A-Za-z0-9@]*)\}")
TODO_RE = re.compile(r"\\todo\{")


def strip_comments(text: str) -> str:
    """Remove LaTeX comments so a commented-out macro is not reported as used."""
    out = []
    for line in text.splitlines():
        idx, escaped = None, False
        for i, ch in enumerate(line):
            if ch == "\\":
                escaped = not escaped
            elif ch == "%" and not escaped:
                idx = i
                break
            else:
                escaped = False
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def resolve_path(paper: Path, target: str) -> Path | None:
    for candidate in (paper / target, paper / f"{target}.tex"):
        if candidate.exists():
            return candidate
    return None


def collect_sources(paper: Path) -> tuple[list[Path], list[str]]:
    """Walk the include graph from ``main.tex``, honouring ``\\IfFileExists`` guards.

    Only files that actually reach the compiled document are checked. A section that is
    guarded out because its results have not been generated is reported as skipped
    rather than as broken -- the guard is the manuscript working as designed.
    """
    root = paper / "main.tex"
    if not root.exists():
        raise SystemExit(f"no main.tex under {paper}")

    ordered: list[Path] = []
    skipped: list[str] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        if path in seen or not path.exists():
            return
        seen.add(path)
        ordered.append(path)
        text = strip_comments(path.read_text(encoding="utf-8"))

        # Any \input appearing after a failing \IfFileExists guard on the same line is
        # unreachable. The manuscript keeps each guard on one line for exactly this
        # reason, so a line-level check is sufficient and predictable.
        for line in text.splitlines():
            unmet = [g for g in IFFILEEXISTS_RE.findall(line)
                     if resolve_path(paper, g) is None]
            for target in INPUT_RE.findall(line):
                child = resolve_path(paper, target)
                if unmet:
                    skipped.append(target)
                    continue
                if child is not None and child.suffix == ".tex" \
                        and "tables/" not in target and "figures/" not in target:
                    visit(child)

    visit(root)
    return ordered, skipped


def check_generated_specials() -> list[str]:
    """Unescaped LaTeX specials in machine-written table cells.

    Tables are emitted with escaping switched off where a header needs math, which
    means every free-text cell has to be escaped by the emitter. When one is missed
    the failure is a hard LaTeX error a long way from its cause, so it is cheaper to
    catch here. Only text outside $...$ is examined.
    """
    problems: list[str] = []
    for path in sorted(TABLES_DIR.glob("*.tex")):
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            # Odd-indexed segments are inside math mode, where _ and ^ are legal.
            outside = line.split("$")[::2]
            for segment in outside:
                for i, ch in enumerate(segment):
                    if ch not in "_^":
                        continue
                    if i and segment[i - 1] == "\\":
                        continue
                    problems.append(
                        f"{path.name}:{lineno}: unescaped {ch!r} outside math mode "
                        f"-- the emitter in visualization/tables.py must escape this "
                        f"cell")
                    break
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", default=str(PAPER_DIR))
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any problem is found")
    args = parser.parse_args(argv)

    paper = Path(args.paper)
    sources, guarded_out = collect_sources(paper)
    if not sources:
        raise SystemExit(f"no manuscript sources found under {paper}")

    macros_file = paper / "generated" / "macros.tex"
    generated = set(NEWCMD_RE.findall(macros_file.read_text(encoding="utf-8"))) \
        if macros_file.exists() else set()

    bib_file = paper / "references.bib"
    bib_keys = set(re.findall(r"@\w+\{([^,]+),",
                              bib_file.read_text(encoding="utf-8"))) \
        if bib_file.exists() else set()

    used_macros: dict[str, list[str]] = {}
    labels: set[str] = set()
    refs: dict[str, list[str]] = {}
    cites: dict[str, list[str]] = {}
    inputs: dict[str, list[str]] = {}
    todos: dict[str, int] = {}

    for path in sources:
        text = strip_comments(path.read_text(encoding="utf-8"))
        name = path.name
        # A \newcommand in the source defines a macro locally; count it as available.
        generated |= set(NEWCMD_RE.findall(text))
        for macro in MACRO_RE.findall(text):
            used_macros.setdefault(macro, []).append(name)
        labels |= set(LABEL_RE.findall(text))
        for ref in REF_RE.findall(text):
            refs.setdefault(ref, []).append(name)
        for group in CITE_RE.findall(text):
            for key in (k.strip() for k in group.split(",")):
                if key:
                    cites.setdefault(key, []).append(name)
        for target in INPUT_RE.findall(text):
            inputs.setdefault(target, []).append(name)
        n_todo = len(TODO_RE.findall(text))
        if n_todo:
            todos[name] = n_todo

    problems = 0

    undefined = sorted(
        m for m in used_macros
        if m not in KNOWN and m not in generated and not m.startswith("@")
    )
    if undefined:
        problems += len(undefined)
        print(f"UNDEFINED MACROS ({len(undefined)}) -- these would fail the build:")
        for macro in undefined:
            where = ", ".join(sorted(set(used_macros[macro])))
            print(f"  \\{macro:44s} used in {where}")
    else:
        print("macros: OK -- every macro used in the prose is defined")

    illegal = sorted(m for m in generated if any(c.isdigit() for c in m))
    if illegal:
        problems += len(illegal)
        print(f"\nILLEGAL MACRO NAMES ({len(illegal)}) -- digits are not permitted in a "
              f"LaTeX control sequence:")
        for macro in illegal:
            print(f"  \\{macro}")

    missing_refs = sorted(r for r in refs if r not in labels)
    if missing_refs:
        problems += len(missing_refs)
        print(f"\nUNRESOLVED REFERENCES ({len(missing_refs)}):")
        for ref in missing_refs:
            print(f"  {ref:40s} referenced from {', '.join(sorted(set(refs[ref])))}")
    else:
        print(f"refs: OK -- all {len(refs)} \\ref targets resolve "
              f"({len(labels)} labels defined)")

    missing_cites = sorted(c for c in cites if c not in bib_keys)
    if missing_cites:
        problems += len(missing_cites)
        print(f"\nMISSING BIBLIOGRAPHY ENTRIES ({len(missing_cites)}):")
        for key in missing_cites:
            print(f"  {key:40s} cited from {', '.join(sorted(set(cites[key])))}")
    else:
        print(f"citations: OK -- all {len(cites)} cited keys exist in references.bib")

    uncited = sorted(bib_keys - set(cites))
    if uncited:
        print(f"\nnote: {len(uncited)} bibliography entries are never cited: "
              f"{', '.join(uncited)}")

    if guarded_out:
        print(f"\nguarded out of this build ({len(guarded_out)}) -- "
              f"\\IfFileExists conditions not met, so the manuscript renders its "
              f"'not available' branch instead:")
        for target in sorted(set(guarded_out)):
            print(f"  {target}")

    missing_inputs = sorted(
        t for t in inputs
        if resolve_path(paper, t) is None
    )
    if missing_inputs:
        problems += len(missing_inputs)
        print(f"\nMISSING \\input TARGETS ({len(missing_inputs)}):")
        for target in missing_inputs:
            print(f"  {target:40s} from {', '.join(sorted(set(inputs[target])))}")
    else:
        print(f"inputs: OK -- all {len(inputs)} \\input targets exist")

    total_todo = sum(todos.values())
    if total_todo:
        print(f"\nTODO markers ({total_todo}) -- author-only content, expected before "
              f"submission but not in it:")
        for name, count in sorted(todos.items()):
            print(f"  {name:32s} {count}")

    words = sum(len(strip_comments(p.read_text(encoding='utf-8')).split())
                for p in sources if p.name != "main.tex")
    print(f"\nscientific body: ~{words} words across {len(sources) - 1} section files")

    specials = check_generated_specials()
    if specials:
        print("\nunescaped LaTeX specials in generated tables "
              f"({len(specials)}) -- these are hard build errors:")
        for line in specials:
            print(f"  {line}")
        problems += len(specials)
    else:
        print("generated tables: OK -- no unescaped specials outside math mode")

    if problems:
        print(f"\n{problems} problem(s) found.")
        return 1 if args.strict else 0
    print("\nNo build-blocking problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
