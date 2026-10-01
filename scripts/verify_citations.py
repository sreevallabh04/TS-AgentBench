#!/usr/bin/env python
"""Verify every citation identifier against arXiv and Crossref.

    python scripts/verify_citations.py
    python scripts/verify_citations.py --strict          # non-zero exit on any failure
    python scripts/verify_citations.py --files paper/references.bib

This is the project's structural defence against fabricated citations. Identifiers are
extracted from ``LITERATURE_REVIEW.md`` and ``paper/references.bib``, resolved against
the public APIs, and the retrieved title is compared to the recorded one. A reference
that does not resolve, or that resolves to a different paper, is reported.

Run this before submission. A hallucinated citation is among the few errors that can
end a paper's life after acceptance, and it is entirely preventable by checking.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import PROJECT_ROOT  # noqa: E402

ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")


def clean_doi(raw: str) -> str:
    """Trim punctuation and publisher path noise from a captured DOI.

    Publisher URLs append an article number after the DOI
    (``.../doi/10.1162/tacl_a_00713/125177/``), and capturing that suffix yields an
    identifier that cannot resolve.

    The trailing segment is stripped **only when the DOI already has a complete
    suffix**, i.e. when more than one slash follows the ``10.NNNN`` prefix. Many valid
    DOI suffixes are entirely numeric (``10.1198/016214506000001437``), so an
    unconditional strip would mangle them into a bare prefix -- which is exactly the
    false alarm this guard exists to prevent.
    """
    doi = raw.rstrip(".,);]}")
    if doi.count("/") >= 2:
        doi = re.sub(r"/\d+$", "", doi)
    return doi


@dataclass
class Citation:
    identifier: str
    kind: str          # "arxiv" | "doi"
    source_file: str
    context: str = ""
    resolved_title: str | None = None
    status: str = "unchecked"


TITLE_RE = re.compile(r"title\s*=\s*[{\"](.+?)[}\"]\s*,?\s*$", re.IGNORECASE | re.DOTALL)


def _extract_bibtex(path: Path) -> list[Citation]:
    """Parse a .bib file entry-wise, pairing each identifier with its entry title.

    Parsing entry-wise matters: in BibTeX the identifier lives on a ``url`` or ``doi``
    line while the title lives on its own. Comparing the resolved title against the
    line the identifier happened to sit on would flag every correct entry as a
    mismatch, which is exactly the failure this function exists to avoid.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    citations: list[Citation] = []
    seen: set[str] = set()

    # Split on entry starts; the first chunk is the file preamble.
    for chunk in re.split(r"\n@", text)[1:]:
        entry = "@" + chunk
        title_match = re.search(r"title\s*=\s*\{(.+?)\}\s*,\s*\n", entry,
                                re.IGNORECASE | re.DOTALL)
        title = " ".join(title_match.group(1).split()) if title_match else ""
        # Strip BibTeX brace-protection so the comparison sees plain words.
        title = title.replace("{", "").replace("}", "")

        for match in list(ARXIV_RE.finditer(entry)) + list(ARXIV_URL_RE.finditer(entry)):
            ident = match.group(1)
            if ident not in seen:
                seen.add(ident)
                citations.append(Citation(ident, "arxiv", path.name, title))
        for match in DOI_RE.finditer(entry):
            ident = clean_doi(match.group(1))
            if ident not in seen:
                seen.add(ident)
                citations.append(Citation(ident, "doi", path.name, title))
    return citations


def extract(path: Path) -> list[Citation]:
    """Pull arXiv ids and DOIs out of a file, paired with the recorded title."""
    if path.suffix == ".bib":
        return _extract_bibtex(path)

    # Markdown and plain text: the identifier and title share a line (a table row).
    text = path.read_text(encoding="utf-8", errors="replace")
    found: dict[str, Citation] = {}
    for line in text.splitlines():
        for match in list(ARXIV_RE.finditer(line)) + list(ARXIV_URL_RE.finditer(line)):
            ident = match.group(1)
            found.setdefault(
                ident, Citation(ident, "arxiv", path.name, line.strip()[:220])
            )
        for match in DOI_RE.finditer(line):
            ident = clean_doi(match.group(1))
            found.setdefault(ident, Citation(ident, "doi", path.name, line.strip()[:220]))
    return list(found.values())


def resolve_arxiv(arxiv_id: str, timeout: int = 20, retries: int = 3) -> str | None:
    """Resolve an arXiv id to its title, retrying transient API failures.

    The arXiv API intermittently returns empty results under load. Without retries a
    perfectly valid citation is reported as unresolved, which would send an author
    hunting for a non-existent problem -- or worse, prompt them to delete a real
    reference.
    """
    for attempt in range(retries):
        title = _resolve_arxiv_once(arxiv_id, timeout)
        if title:
            return title
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    return None


def _resolve_arxiv_once(arxiv_id: str, timeout: int = 20) -> str | None:
    import requests

    try:
        response = requests.get(
            "http://export.arxiv.org/api/query",
            params={"id_list": arxiv_id, "max_results": 1},
            timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        title = entry.find("a:title", ns)
        if title is None or title.text is None:
            return None
        # arXiv's API rejects an id that does not exist by echoing an error entry.
        if "Error" in (title.text or ""):
            return None
        return " ".join(title.text.split())
    except Exception:
        return None


def resolve_doi(doi: str, timeout: int = 20) -> str | None:
    import requests

    try:
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=timeout,
            headers={"User-Agent": "TS-AgentBench citation verifier"},
        )
        if response.status_code != 200:
            return None
        titles = response.json().get("message", {}).get("title") or []
        return " ".join(titles[0].split()) if titles else None
    except Exception:
        return None


def _normalise(text: str) -> str:
    """Lowercase, strip markup and collapse non-alphanumerics for comparison."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def title_matches(resolved: str, recorded: str) -> bool:
    """Does the recorded context contain, or closely match, the resolved title?

    Two checks, because the recorded context has two shapes. In a .bib file it is the
    entry's title field, where a fuzzy ratio is right. In the literature review it is a
    whole markdown table row containing the title plus contribution and limitation
    prose, where a ratio would be low even for a perfect match -- so containment of the
    title's leading words is the correct test there.
    """
    a, b = _normalise(resolved), _normalise(recorded)
    if not a or not b:
        return False
    if a in b:
        return True
    # Leading words are the distinctive part of a title; subtitles vary between venues.
    head = " ".join(a.split()[:6])
    if head and head in b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.60


def _safe(text: str) -> str:
    """Make a string printable on a legacy Windows console."""
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files", nargs="*",
        default=["LITERATURE_REVIEW.md", "paper/references.bib"],
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between API calls, to stay polite")
    args = parser.parse_args(argv)

    citations: list[Citation] = []
    for name in args.files:
        path = PROJECT_ROOT / name
        if not path.exists():
            print(f"  (skipping missing file: {name})")
            continue
        citations += extract(path)

    if not citations:
        print("No citation identifiers found.")
        return 0

    print(f"Verifying {len(citations)} identifiers...\n")
    resolved = unresolved = mismatched = 0

    for citation in citations:
        title = (
            resolve_arxiv(citation.identifier)
            if citation.kind == "arxiv"
            else resolve_doi(citation.identifier)
        )
        time.sleep(args.delay)

        if title is None:
            citation.status = "UNRESOLVED"
            unresolved += 1
            print(f"  [UNRESOLVED] {citation.kind}:{citation.identifier}")
            print(f"               context: {_safe(citation.context)[:150]}")
            continue

        citation.resolved_title = title
        if not title_matches(title, citation.context):
            citation.status = "TITLE MISMATCH"
            mismatched += 1
            print(f"  [MISMATCH]   {citation.kind}:{citation.identifier}")
            print(f"               recorded: {_safe(citation.context)[:150]}")
            print(f"               resolved: {_safe(title)}")
        else:
            citation.status = "ok"
            resolved += 1
            print(f"  [ok]         {citation.identifier}  {_safe(title)[:72]}")

    print(
        f"\n{resolved} verified, {mismatched} title mismatches, "
        f"{unresolved} unresolved, out of {len(citations)}."
    )
    if mismatched or unresolved:
        print(
            "\nEvery unresolved or mismatched entry must be corrected or removed "
            "before submission. Do not cite an identifier that does not resolve."
        )
    return 1 if args.strict and (mismatched or unresolved) else 0


if __name__ == "__main__":
    raise SystemExit(main())
