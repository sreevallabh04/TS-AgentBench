#!/usr/bin/env python
"""Fill missing author fields from the source record, never from memory.

    python scripts/fill_authors.py            # report what would change
    python scripts/fill_authors.py --write    # write it

Bibliography entries without an ``author`` field render as the citation key's stub
("fla 2025"), which reads as unfinished and makes a reference impossible to check. This
fetches the author list from arXiv (Atom API) or Crossref using the identifier already
recorded in the entry, and writes it in BibTeX form. Entries with no resolvable
identifier are reported and left untouched: an author list that cannot be traced to a
source is not written.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import PAPER_DIR  # noqa: E402

BIB = PAPER_DIR / "references.bib"
MAX_AUTHORS = 12  # beyond this, "and others" -- the style truncates long lists anyway


def arxiv_authors(arxiv_id: str, retries: int = 3) -> list[str] | None:
    import requests
    for attempt in range(retries):
        try:
            r = requests.get("http://export.arxiv.org/api/query",
                             params={"id_list": arxiv_id, "max_results": 1}, timeout=20)
            r.raise_for_status()
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entry = ET.fromstring(r.text).find("a:entry", ns)
            if entry is not None:
                names = [a.findtext("a:name", default="", namespaces=ns).strip()
                         for a in entry.findall("a:author", ns)]
                names = [n for n in names if n]
                if names:
                    return names
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def crossref_authors(doi: str) -> list[str] | None:
    import requests
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20,
                         headers={"User-Agent": "TS-AgentBench author filler"})
        if r.status_code != 200:
            return None
        people = r.json().get("message", {}).get("author") or []
        names = [f"{p.get('given', '')} {p.get('family', '')}".strip() for p in people]
        return [n for n in names if n] or None
    except Exception:
        return None


def to_bibtex(names: list[str]) -> str:
    """'Given Family' -> 'Family, Given', joined with ' and ', long lists truncated."""
    def flip(n: str) -> str:
        parts = n.split()
        return n if len(parts) < 2 else f"{parts[-1]}, {' '.join(parts[:-1])}"
    flipped = [flip(n) for n in names[:MAX_AUTHORS]]
    if len(names) > MAX_AUTHORS:
        flipped.append("others")
    return " and\n               ".join(flipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    text = BIB.read_text(encoding="utf-8")
    entries = list(re.finditer(r"(@\w+\{([^,]+),)(.*?)(\n\})", text, re.S))
    updated, skipped = [], []
    for m in reversed(entries):  # reversed so earlier offsets stay valid while editing
        key, body = m.group(2), m.group(3)
        if re.search(r"^\s*author\s*=", body, re.M):
            continue
        ax = re.search(r"arxiv\.org/abs/([\d.]+v?\d*)", body)
        doi = re.search(r"(10\.\d{4,}/[^\s,}]+)", body)
        names, source = None, None
        if ax:
            names, source = arxiv_authors(ax.group(1)), f"arXiv:{ax.group(1)}"
        elif doi:
            names, source = crossref_authors(doi.group(1)), f"doi:{doi.group(1)}"
        if not names:
            skipped.append((key, source or "no arXiv id or DOI in entry"))
            continue
        field = f"\n  author  = {{{to_bibtex(names)}}},"
        new_entry = m.group(1) + field + body + m.group(4)
        text = text[:m.start()] + new_entry + text[m.end():]
        updated.append((key, source, len(names), names[0]))
        time.sleep(0.5)

    for key, source, n, first in reversed(updated):
        print(f"  filled  {key:24s} {n:3d} authors from {source}  (first: {first})")
    for key, why in skipped:
        print(f"  SKIPPED {key:24s} {why}")
    if args.write and updated:
        BIB.write_text(text, encoding="utf-8")
        print(f"\nwrote {len(updated)} author fields to {BIB}")
    elif updated:
        print("\n(dry run; pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
