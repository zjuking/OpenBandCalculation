#!/usr/bin/env python3
"""
Assemble the CPC submission package.

    python paper/cpc/make_submission.py

Writes paper/cpc/dist/:

    README-upload.md      what goes where in Editorial Manager
    manuscript/           the files uploaded individually (tex, bib, figures, pdf,
                          highlights, cover letter)
    obpds-2.0.zip         the program archive (source, tests, example run,
                          reproduction scripts, licence)

The program archive is the repository at submission time, minus build
artefacts and the virtual environment.  Nothing is downloaded or invented: the
zip is built from what is on disk, so if a file is missing from the archive it
was missing from the repository.
"""
from __future__ import annotations

import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DIST = os.path.join(HERE, "dist")

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".workbuddy",
                ".workbuddy-ai", "dist", ".pytest_cache", ".mypy_cache"}
# `paper/cpc/data/*.out` are verification logs and must stay, so ".out" is
# deliberately absent here; the LaTeX transcript is dropped by name below.
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".aux", ".blg", ".log", ".fls",
                  ".fdb_latexmk", ".synctex.gz"}
EXCLUDE_NAMES = {"manuscript.log", "manuscript.aux", "manuscript.blg",
                 "manuscript.fls", "manuscript.fdb_latexmk",
                 "manuscript.synctex.gz"}

MANUSCRIPT_FILES = ["manuscript.tex", "refs.bib", "manuscript.pdf",
                    "highlights.txt", "cover_letter.md"]


def keep(path, name):
    if name in EXCLUDE_NAMES:
        return False
    if os.path.splitext(name)[1] in EXCLUDE_SUFFIX:
        return False
    return True


def main():
    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)

    # ---- 1. the files uploaded individually --------------------------------
    mdir = os.path.join(DIST, "manuscript")
    os.makedirs(mdir)
    missing = []
    for f in MANUSCRIPT_FILES:
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, mdir)
        else:
            missing.append(f)
    shutil.copytree(os.path.join(HERE, "figures"), os.path.join(mdir, "figures"),
                    ignore=shutil.ignore_patterns("*.png"))

    # ---- 2. the program archive -------------------------------------------
    zpath = os.path.join(DIST, "obpds-2.0.zip")
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                if not keep(full, fn):
                    continue
                z.write(full, os.path.join("obpds-2.0", rel))
                n += 1

    # ---- 3. upload notes ---------------------------------------------------
    notes = f"""# What to upload where (Editorial Manager, comphy)

## Item type "Manuscript"
`manuscript/manuscript.pdf` (a compiled copy; the source is below)

## Item type "LaTeX source" / "Supplementary material"
`manuscript/manuscript.tex`, `manuscript/refs.bib`, `manuscript/figures/` (6 PDFs)

## Item type "Highlights"
`manuscript/highlights.txt` (5 bullets, each under 85 characters)

## Cover letter
paste or upload `manuscript/cover_letter.md`

## Item type "Program files"
`obpds-2.0.zip` --- {n} files, built from the repository as it stands.
Contains the source (`src/`), the four verification suites (`tests/`), the
comprehensive example run and its expected output (`examples/`), the licence,
the README, and the scripts that regenerate every figure and table of the
manuscript (`paper/cpc/`). `paper/cpc/data/environment.txt` records the exact
interpreter and library versions behind the timings, and
`paper/cpc/data/bench.json` holds the raw measurements.

## Before pressing submit
- [ ] replace `https://github.com/<owner>/obpds` in the Program Summary
- [ ] replace the Zenodo DOI in Data availability, or delete the sentence
- [ ] author names, affiliation and e-mail in manuscript.tex
- [ ] the licence question in the cover letter: AGPL-3.0 is not on the CPC
      approved list and cannot be changed while the program derives from OBPDS
"""
    with open(os.path.join(DIST, "README-upload.md"), "w", encoding="utf-8") as f:
        f.write(notes)

    size = os.path.getsize(zpath) / 1e6
    print(f"dist/README-upload.md")
    print(f"dist/manuscript/  ({len(MANUSCRIPT_FILES)} files + figures/)")
    print(f"dist/obpds-2.0.zip  {n} files, {size:.1f} MB")
    if missing:
        print("MISSING:", ", ".join(missing))


if __name__ == "__main__":
    main()
